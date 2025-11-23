import os
import re
import numpy as np
import torch
from torch.utils.data import DataLoader
import flwr as fl
from flwr.common import Context, parameters_to_ndarrays
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import itertools
from model import ConvLSTMClassifier
from prepare_data import SequenceFeatureDataset
from config import (
    FED_ROUNDS,
    BATCH_SIZE,
    LEARNING_RATE,
    DEVICE,
    LOCAL_EPOCHS,
    FED_DIR,
)
from typing import List, Tuple, Dict, Any





class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model: torch.nn.Module, dataset: SequenceFeatureDataset):
        self.model = model.to(DEVICE)
        self.dataset = dataset
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=LEARNING_RATE)

    def get_parameters(self, config=None) -> List[np.ndarray]:
        return [p.detach().cpu().numpy() for p in self.model.parameters()]

    def set_parameters(self, parameters: List[np.ndarray]):
        params = list(self.model.parameters())
        if len(params) != len(parameters):
            raise ValueError(
                f"Got {len(parameters)} parameter arrays but model expects {len(params)} parameters."
            )
        for p, new in zip(params, parameters):
            arr = np.asarray(new)
            t = torch.from_numpy(arr).to(p.data.device)
            if t.dtype != p.data.dtype:
                t = t.to(p.data.dtype)
            p.data.copy_(t)

    def fit(self, parameters, config):
        # Load server parameters
        self.set_parameters(parameters)

        loader = DataLoader(self.dataset, batch_size=BATCH_SIZE, shuffle=True)
        self.model.train()

        local_epochs = LOCAL_EPOCHS

        total_loss = 0.0
        total_examples = 0

        print(f"[client] start training for {local_epochs} local epochs on {len(self.dataset)} examples.")
        for _ in range(local_epochs):
            for seq, feat, label in loader:
                seq = seq.to(DEVICE)
                feat = feat.to(DEVICE)
                label = label.to(DEVICE)

                self.optimizer.zero_grad()
                logits = self.model(seq, feat)
                loss = self.criterion(logits, label)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item() * seq.size(0)
                total_examples += seq.size(0)

        avg_loss = (total_loss / total_examples) if total_examples > 0 else 0.0

        # Return parameters, number of examples, and fit metrics
        return self.get_parameters(), len(self.dataset), {"loss": float(avg_loss)}

    def evaluate(self, parameters, config):
        # Load parameters
        self.set_parameters(parameters)

        loader = DataLoader(self.dataset, batch_size=BATCH_SIZE)
        self.model.eval()
        ys_list, ypred_list = [], []

        total_loss = 0.0
        total_examples = 0

        with torch.no_grad():
            for seq, feat, label in loader:
                seq = seq.to(DEVICE)
                feat = feat.to(DEVICE)
                label = label.to(DEVICE)

                logits = self.model(seq, feat)
                loss = self.criterion(logits, label)

                preds = logits.argmax(dim=1).cpu().numpy()
                ys = label.cpu().numpy()

                ypred_list.append(preds)
                ys_list.append(ys)

                total_loss += float(loss.item()) * seq.size(0)
                total_examples += seq.size(0)

        if total_examples > 0:
            avg_loss = total_loss / total_examples
            ys_all = np.concatenate(ys_list)
            ypred_all = np.concatenate(ypred_list)
            acc = float(np.mean(ys_all == ypred_all))
            try:
                f1 = float(f1_score(ys_all, ypred_all, average="macro"))
            except Exception:
                f1 = 0.0
        else:
            avg_loss = 0.0
            acc = 0.0
            f1 = 0.0

        return float(avg_loss), len(self.dataset), {"acc": acc, "f1": f1}



def _get_client_index_from_context(context: Context, n_clients: int) -> int:
    node_config = getattr(context, "node_config", {}) or {}
    for key in ("partition-id", "partition_id", "partitionId", "partition"):
        if key in node_config:
            try:
                cid_i = int(node_config[key])
            except Exception:
                raise ValueError(f"partition id in node_config is not an integer: {node_config[key]}")
            if cid_i < 0 or cid_i >= n_clients:
                raise ValueError(f"partition id {cid_i} out of range for {n_clients} clients.")
            return cid_i

    node_id = getattr(context, "node_id", None)
    if node_id is None:
        raise ValueError("Context has no node_id and node_config contains no partition id.")

    numbers = re.findall(r"\d+", str(node_id))
    if not numbers:
        raise ValueError(f"Could not extract numeric client index from node_id '{node_id}'")
    cid_i = int(numbers[-1])
    if cid_i < 0 or cid_i >= n_clients:
        raise ValueError(f"Extracted client index {cid_i} from node_id '{node_id}' is out of range for {n_clients} clients.")
    return cid_i


def make_client_fn(client_datasets: List[SequenceFeatureDataset], feat_dim: int, num_classes: int):
    n_clients = len(client_datasets)

    def client_fn(context: Context):
        cid_i = _get_client_index_from_context(context, n_clients)
        dataset = client_datasets[cid_i]
        model = ConvLSTMClassifier(seq_channels=3, feat_dim=feat_dim, num_classes=num_classes)
        return FlowerClient(model, dataset).to_client()

    return client_fn


# metric aggregation functions
def aggregate_fit_metrics(results: List[Tuple[int, Dict[str, Any]]]) -> Dict[str, float]:
    total_examples = 0
    weighted_loss = 0.0
    for num_examples, metrics in results:
        if metrics is None:
            continue
        loss = metrics.get("loss")
        if loss is None:
            continue
        n = int(num_examples)
        total_examples += n
        weighted_loss += float(loss) * n
    if total_examples == 0:
        return {}
    return {"loss": weighted_loss / total_examples}


def aggregate_eval_metrics(results: List[Tuple[int, Dict[str, Any]]]) -> Dict[str, float]:
    total_examples = 0
    weighted_acc = 0.0
    weighted_f1 = 0.0
    for num_examples, metrics in results:
        if metrics is None:
            continue
        acc = metrics.get("acc")
        f1 = metrics.get("f1")
        if acc is None or f1 is None:
            continue
        n = int(num_examples)
        total_examples += n
        weighted_acc += float(acc) * n
        weighted_f1 += float(f1) * n
    if total_examples == 0:
        return {}
    return {"acc": weighted_acc / total_examples, "f1": weighted_f1 / total_examples}



# Plots
def plot_metrics(history: Dict[str, List[Tuple[int, float]]], out_dir: str):
    """
    history: dict like {"loss": [(round, value), ...], "acc": [...], "f1":[...]}
    Saves PNG files for loss, acc, f1.
    """
    rounds_loss = [r for r, _ in history.get("loss", [])]
    vals_loss = [v for _, v in history.get("loss", [])]

    rounds_acc = [r for r, _ in history.get("acc", [])]
    vals_acc = [v for _, v in history.get("acc", [])]

    rounds_f1 = [r for r, _ in history.get("f1", [])]
    vals_f1 = [v for _, v in history.get("f1", [])]

    # Loss plot
    if vals_loss:
        plt.figure()
        plt.plot(rounds_loss, vals_loss, marker="o")
        plt.xlabel("Round")
        plt.ylabel("Loss (eval / weighted)")
        plt.title("Federated learning: eval loss per round")
        plt.grid(True)
        loss_path = os.path.join(out_dir, "eval_loss_per_round.png")
        plt.savefig(loss_path, bbox_inches="tight")
        plt.close()
        print(f"Saved loss plot: {loss_path}")

    # Accuracy plot
    if vals_acc:
        plt.figure()
        plt.plot(rounds_acc, vals_acc, marker="o")
        plt.xlabel("Round")
        plt.ylabel("Accuracy (weighted)")
        plt.title("Federated learning: accuracy per round")
        plt.grid(True)
        acc_path = os.path.join(out_dir, "accuracy_per_round.png")
        plt.savefig(acc_path, bbox_inches="tight")
        plt.close()
        print(f"Saved accuracy plot: {acc_path}")

    # F1 plot
    if vals_f1:
        plt.figure()
        plt.plot(rounds_f1, vals_f1, marker="o")
        plt.xlabel("Round")
        plt.ylabel("F1 score (macro, weighted)")
        plt.title("Federated learning: F1 (macro) per round")
        plt.grid(True)
        f1_path = os.path.join(out_dir, "f1_per_round.png")
        plt.savefig(f1_path, bbox_inches="tight")
        plt.close()
        print(f"Saved f1 plot: {f1_path}")


def plot_confusion_matrix(cm: np.ndarray, classes: List[str], out_path: str, normalize: bool = True):
    """
    cm: confusion matrix (num_classes x num_classes)
    classes: class labels (strings)
    """
    if normalize:
        with np.errstate(all="ignore"):
            cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
            cm_normalized = np.nan_to_num(cm_normalized)
    else:
        cm_normalized = cm

    plt.figure(figsize=(6, 6))
    plt.imshow(cm_normalized, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion matrix (normalized)" if normalize else "Confusion matrix")
    plt.colorbar(fraction=0.046, pad=0.04)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, ha="right")
    plt.yticks(tick_marks, classes)

    thresh = cm_normalized.max() / 2.0 if cm_normalized.size > 0 else 0.5
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            f"{cm_normalized[i, j]:.2f}",
            horizontalalignment="center",
            color="white" if cm_normalized[i, j] > thresh else "black",
            fontsize=8,
        )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved confusion matrix: {out_path}")


# simulation : recrod metrics per round
def run_federated_simulation(client_datasets, feat_dim, num_classes):

    history_metrics = {"loss": [], "acc": [], "f1": []}
    final_model_ndarrays = None

    # Custom FedAvg strategy to hook aggregation results each round
    class HookedFedAvg(fl.server.strategy.FedAvg):
        def aggregate_fit(self, rnd, results, failures):
            # Call parent to do aggregation of parameters and fit metrics
            aggregated = super().aggregate_fit(rnd, results, failures)
            if aggregated is not None:
                # aggregated is a tuple (parameters, metrics) or similar
                try:
                    params, metrics = aggregated
                except Exception:
                    # Some Flower versions return only parameters
                    params = aggregated[0] if isinstance(aggregated, (list, tuple)) else aggregated
                    metrics = {}
                # metrics here are server-side aggregated fit metrics (loss) if provided by fit_metrics_aggregation_fn
                if isinstance(metrics, dict):
                    loss = metrics.get("loss")
                    if loss is not None:
                        history_metrics["loss"].append((rnd, float(loss)))
                # capture parameters
                nonlocal final_model_ndarrays
                final_model_ndarrays = params
            return aggregated

        def aggregate_evaluate(self, rnd, results, failures):
            # Call parent to aggregate evaluate results
            aggregated = super().aggregate_evaluate(rnd, results, failures)
            if aggregated is not None:
                try:
                    params, metrics = aggregated
                except Exception:
                    # Older Flower may return (metrics) or dict
                    params = None
                    metrics = aggregated
                # metrics should contain aggregated eval metrics via evaluate_metrics_aggregation_fn
                if isinstance(metrics, dict):
                    acc = metrics.get("acc")
                    f1 = metrics.get("f1")
                    if acc is not None:
                        history_metrics["acc"].append((rnd, float(acc)))
                    if f1 is not None:
                        history_metrics["f1"].append((rnd, float(f1)))
            return aggregated

    
    strategy = HookedFedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=len(client_datasets),
        min_evaluate_clients=len(client_datasets),
        min_available_clients=len(client_datasets),
        fit_metrics_aggregation_fn=aggregate_fit_metrics,
        evaluate_metrics_aggregation_fn=aggregate_eval_metrics,
    )

    client_fn = make_client_fn(client_datasets, feat_dim, num_classes)

    
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=len(client_datasets),
        config=fl.server.ServerConfig(num_rounds=FED_ROUNDS),
        strategy=strategy,
    )

    # save final model
    if final_model_ndarrays is None:
        raise RuntimeError("No final model parameters were captured from the strategy.")

    # Save final global model
    model = ConvLSTMClassifier(seq_channels=3, feat_dim=feat_dim, num_classes=num_classes).to(DEVICE)
    for p, w in zip(model.parameters(), parameters_to_ndarrays(final_model_ndarrays)):
        t = torch.tensor(w, dtype=p.data.dtype, device=p.data.device)
        p.data.copy_(t)
    final_model_path = os.path.join(FED_DIR, "federated_model.pt")
    torch.save(model.state_dict(), final_model_path)
    print(f"✔ Saved final federated model to {final_model_path}")


    flattened = []
    for cd in client_datasets:
        flattened.extend(list(cd))  # each cd yields tuples (seq, feat, label)
    if len(flattened) == 0:
        raise RuntimeError("No data found across client datasets to evaluate final model.")

    full_loader = DataLoader(flattened, batch_size=64, shuffle=False)

    # Evaluate model on full dataset and compute confusion matrix, acc, f1, loss
    model.eval()
    ys_all = []
    ypred_all = []
    total_loss = 0.0
    total_examples = 0
    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for seq, feat, label in full_loader:
            seq = seq.to(DEVICE)
            feat = feat.to(DEVICE)
            label = label.to(DEVICE)

            logits = model(seq, feat)
            loss = criterion(logits, label)

            preds = logits.argmax(dim=1).cpu().numpy()
            ys = label.cpu().numpy()

            ypred_all.append(preds)
            ys_all.append(ys)

            total_loss += float(loss.item()) * seq.size(0)
            total_examples += seq.size(0)

    ys_all = np.concatenate(ys_all)
    ypred_all = np.concatenate(ypred_all)
    eval_loss = total_loss / total_examples if total_examples > 0 else 0.0
    eval_acc = float(accuracy_score(ys_all, ypred_all))
    try:
        eval_f1 = float(f1_score(ys_all, ypred_all, average="macro"))
    except Exception:
        eval_f1 = 0.0

    print("Final model evaluated on full dataset — loss: {:.4f}, acc: {:.4f}, f1: {:.4f}".format(eval_loss, eval_acc, eval_f1))

    # Confusion matrix
    cm = confusion_matrix(ys_all, ypred_all)

    class_names = None
    try:
        maybe_le = getattr(client_datasets[0], "label_encoder", None)
        if maybe_le is not None and hasattr(maybe_le, "classes_"):
            class_names = [str(c) for c in maybe_le.classes_]
    except Exception:
        class_names = None

    if class_names is None:
        n_classes = cm.shape[0]
        class_names = [str(i) for i in range(n_classes)]

    cm_path = os.path.join(FED_DIR, "confusion_matrix_full.png")
    plot_confusion_matrix(cm, class_names, cm_path, normalize=True)

    plot_history = {}
    for k in ("loss", "acc", "f1"):
        plot_history[k] = sorted(history_metrics.get(k, []), key=lambda x: x[0])

    plot_metrics(plot_history, FED_DIR)

    np.savez_compressed(os.path.join(FED_DIR, "federated_history.npz"),
                        loss=np.array(plot_history.get("loss", [])),
                        acc=np.array(plot_history.get("acc", [])),
                        f1=np.array(plot_history.get("f1", [])))
    print(f"Saved aggregated history to {os.path.join(FED_DIR, 'federated_history.npz')}")

    return model, {"loss": eval_loss, "acc": eval_acc, "f1": eval_f1, "confusion_matrix_path": cm_path}
