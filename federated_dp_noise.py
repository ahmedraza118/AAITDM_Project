import os
import re
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
import flwr as fl
from flwr.common import Context, parameters_to_ndarrays, ndarrays_to_parameters
from flwr.common.typing import EvaluateRes
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Any

from model import ConvLSTMClassifier
from prepare_data import SequenceFeatureDataset
from config import (
    FED_ROUNDS,
    BATCH_SIZE,
    LEARNING_RATE,
    DEVICE,
    LOCAL_EPOCHS,
    FED_DIR2,
    DP_ENABLED,
    DP_CLIP_NORM,
    DP_NOISE_MULTIPLIER,
    DP_DELTA,
    POISONING_ENABLED,
    POISON_CLIENT_FRACTION,
    LABEL_NOISE_RATE,
)

os.makedirs(FED_DIR2, exist_ok=True)

# Byzantine Robust Aggregation
# ============================================================

def trimmed_mean(param_list, trim_ratio=0.2):
    stacked = np.stack(param_list)
    k = int(trim_ratio * stacked.shape[0])
    stacked = np.sort(stacked, axis=0)
    return np.mean(stacked[k:-k], axis=0)

# Client
# =======

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, dataset, cid, poison=False):
        self.model = model.to(DEVICE)
        self.dataset = dataset
        self.cid = cid
        self.poison = poison
        self.grad_norms = []

        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=LEARNING_RATE)

    def get_parameters(self, config=None):
        return [p.detach().cpu().numpy() for p in self.model.parameters()]

    def set_parameters(self, parameters):
        for p, w in zip(self.model.parameters(), parameters):
            p.data.copy_(torch.tensor(w, device=p.device, dtype=p.dtype))

    def _maybe_poison(self, y):
        if not self.poison:
            return y
        mask = torch.rand_like(y.float()) < LABEL_NOISE_RATE
        rand = torch.randint(0, y.max() + 1, y.shape, device=y.device)
        return torch.where(mask, rand, y)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        loader = DataLoader(self.dataset, batch_size=BATCH_SIZE, shuffle=True)

        total_loss, total_examples = 0.0, 0

        for _ in range(LOCAL_EPOCHS):
            for seq, feat, label in loader:
                seq, feat, label = seq.to(DEVICE), feat.to(DEVICE), label.to(DEVICE)
                label = self._maybe_poison(label)

                self.optimizer.zero_grad()
                logits = self.model(seq, feat)
                loss = self.criterion(logits, label)
                loss.backward()

                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), DP_CLIP_NORM
                )
                self.grad_norms.append(float(grad_norm))

                if DP_ENABLED:
                    for p in self.model.parameters():
                        if p.grad is not None:
                            p.grad += torch.normal(
                                0,
                                DP_NOISE_MULTIPLIER * DP_CLIP_NORM,
                                p.grad.shape,
                                device=p.grad.device,
                            )

                self.optimizer.step()

                total_loss += loss.item() * seq.size(0)
                total_examples += seq.size(0)

        np.save(os.path.join(FED_DIR2, f"grad_norms_client_{self.cid}.npy"),
                np.array(self.grad_norms))

        return self.get_parameters(), len(self.dataset), {
            "loss": total_loss / max(total_examples, 1),
            "cid": self.cid,
            "poisoned": self.poison,
        }

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loader = DataLoader(self.dataset, batch_size=BATCH_SIZE)

        ys, preds = [], []
        total_loss, total_examples = 0.0, 0

        with torch.no_grad():
            for seq, feat, label in loader:
                seq, feat, label = seq.to(DEVICE), feat.to(DEVICE), label.to(DEVICE)
                logits = self.model(seq, feat)
                loss = self.criterion(logits, label)

                total_loss += loss.item() * seq.size(0)
                total_examples += seq.size(0)

                preds.append(logits.argmax(1).cpu().numpy())
                ys.append(label.cpu().numpy())

        ys = np.concatenate(ys)
        preds = np.concatenate(preds)

        acc = accuracy_score(ys, preds)
        f1 = f1_score(ys, preds, average="macro")

        cm = confusion_matrix(ys, preds)
        np.save(os.path.join(FED_DIR2, f"cm_client_{self.cid}.npy"), cm)

        return total_loss / max(total_examples, 1), len(self.dataset), {
            "acc": acc,
            "f1": f1,
            "cid": self.cid,
            "poisoned": self.poison,
        }

# CID

def _get_client_index_from_context(context: Context, n_clients: int) -> int:
    node_config = getattr(context, "node_config", {}) or {}
    for key in ("partition-id", "partition_id", "partitionId", "partition"):
        if key in node_config:
            return int(node_config[key])
    numbers = re.findall(r"\d+", str(context.node_id))
    return int(numbers[-1])


# Aggregation metric

def aggregate_fit_metrics(results):
    tot, loss = 0, 0.0
    for n, m in results:
        if m:
            loss += m["loss"] * n
            tot += n
    return {"loss": loss / max(tot, 1)}

def aggregate_eval_metrics(results):
    tot, acc, f1 = 0, 0.0, 0.0
    for n, m in results:
        if m:
            acc += m["acc"] * n
            f1 += m["f1"] * n
            tot += n
    return {"acc": acc / tot, "f1": f1 / tot}



def make_client_fn(client_datasets, feat_dim, num_classes):
    n = len(client_datasets)
    poisoned = set(
        np.random.choice(range(n), int(n * POISON_CLIENT_FRACTION), replace=False)
    ) if POISONING_ENABLED else set()

    def client_fn(context: Context):
        cid = _get_client_index_from_context(context, n)
        model = ConvLSTMClassifier(3, feat_dim, num_classes)
        return FlowerClient(
            model,
            client_datasets[cid],
            cid,
            cid in poisoned,
        ).to_client()

    return client_fn

# start simulation

def run_federated_simulation(client_datasets, feat_dim, num_classes):

    history = {
        "fedavg_acc": [],
        "trimmed_acc": [],
        "client_metrics": [],
    }

    # -- FedAvg --
    fedavg = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=len(client_datasets),
        min_evaluate_clients=len(client_datasets),
        min_available_clients=len(client_datasets),
        fit_metrics_aggregation_fn=aggregate_fit_metrics,
        evaluate_metrics_aggregation_fn=aggregate_eval_metrics,
    )

    fl.simulation.start_simulation(
        client_fn=make_client_fn(client_datasets, feat_dim, num_classes),
        num_clients=len(client_datasets),
        config=fl.server.ServerConfig(num_rounds=FED_ROUNDS),
        strategy=fedavg,
    )

    # -- Trimmed Mean --
    class TrimmedMeanStrategy(fl.server.strategy.FedAvg):
        def aggregate_fit(self, rnd, results, failures):
            params = [parameters_to_ndarrays(res.parameters) for _, res in results]
            agg = [
                trimmed_mean([p[i] for p in params])
                for i in range(len(params[0]))
            ]
            return ndarrays_to_parameters(agg), {}

    trimmed = TrimmedMeanStrategy(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=len(client_datasets),
        min_evaluate_clients=len(client_datasets),
        min_available_clients=len(client_datasets),
        fit_metrics_aggregation_fn=aggregate_fit_metrics,
        evaluate_metrics_aggregation_fn=aggregate_eval_metrics,
    )

    fl.simulation.start_simulation(
        client_fn=make_client_fn(client_datasets, feat_dim, num_classes),
        num_clients=len(client_datasets),
        config=fl.server.ServerConfig(num_rounds=FED_ROUNDS),
        strategy=trimmed,
    )

    # -- Fairness metrics --
    accs = []
    poisoned_accs = []
    clean_accs = []

    for f in os.listdir(FED_DIR2):
        if f.startswith("cm_client_"):
            cid = int(f.split("_")[-1].replace(".npy", ""))
            cm = np.load(os.path.join(FED_DIR2, f))
            acc = np.trace(cm) / cm.sum()
            accs.append(acc)

    fairness = {
        "mean_acc": float(np.mean(accs)),
        "std_acc": float(np.std(accs)),
        "gap": float(np.max(accs) - np.min(accs)),
    }

    with open(os.path.join(FED_DIR2, "fairness_metrics.json"), "w") as f:
        json.dump(fairness, f, indent=2)

    # -- Overlay grad norms --
    plt.figure()
    for f in os.listdir(FED_DIR2):
        if f.startswith("grad_norms_client_"):
            g = np.load(os.path.join(FED_DIR2, f))
            plt.hist(g, bins=40, alpha=0.4, density=True)

    plt.xlabel("Gradient Norm")
    plt.ylabel("Density")
    plt.title("Overlay of Gradient Norms Across Clients")
    plt.savefig(os.path.join(FED_DIR2, "grad_norms_overlay.png"))
    plt.close()

    print("✔ Federated simulation completed with all extended metrics.")
