import os
import json
import numpy as np
import matplotlib.pyplot as plt
import itertools

from config import FED_DIR2


# -------------------------------------------------
# Confusion Matrix Plot
# -------------------------------------------------
def plot_confusion_matrix(cm, title, out_path, normalize=True):
    if normalize:
        with np.errstate(all="ignore"):
            cm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
            cm = np.nan_to_num(cm)

    plt.figure(figsize=(5, 5))
    plt.imshow(cm, cmap="Blues")
    plt.title(title)
    plt.colorbar()

    ticks = np.arange(cm.shape[0])
    plt.xticks(ticks)
    plt.yticks(ticks)

    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            f"{cm[i, j]:.2f}",
            ha="center",
            va="center",
            color="white" if cm[i, j] > thresh else "black",
            fontsize=8,
        )

    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")


# -------------------------------------------------
# Overlay Gradient Norms
# -------------------------------------------------
def plot_grad_norms_overlay(grad_files):
    plt.figure()
    for f in grad_files:
        norms = np.load(os.path.join(FED_DIR2, f))
        plt.hist(norms, bins=40, alpha=0.4, density=True)

    plt.xlabel("Gradient Norm")
    plt.ylabel("Density")
    plt.title("Overlay of Gradient Norm Distributions (All Clients)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(FED_DIR2, "grad_norms_overlay.png"))
    plt.close()
    print("Saved: grad_norms_overlay.png")


# -------------------------------------------------
# Fairness Metrics Plot (JSON)
# -------------------------------------------------
def plot_fairness_metrics(json_path):
    with open(json_path, "r") as f:
        fairness = json.load(f)

    labels = ["Mean Accuracy", "Std Dev", "Max–Min Gap"]
    values = [
        fairness["mean_acc"],
        fairness["std_acc"],
        fairness["gap"],
    ]

    plt.figure()
    plt.bar(labels, values)
    plt.ylabel("Value")
    plt.title("Client Fairness Metrics")
    plt.grid(axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(FED_DIR2, "fairness_metrics.png"))
    plt.close()
    print("Saved: fairness_metrics.png")


# -------------------------------------------------
# Client-wise Accuracy Bar Plot
# -------------------------------------------------
def plot_client_accuracy_from_cm(cm_files):
    accs = []
    cids = []

    for f in cm_files:
        cid = int(f.split("_")[-1].replace(".npy", ""))
        cm = np.load(os.path.join(FED_DIR2, f))
        acc = np.trace(cm) / cm.sum()
        accs.append(acc)
        cids.append(cid)

    plt.figure()
    plt.bar(cids, accs)
    plt.xlabel("Client ID")
    plt.ylabel("Accuracy")
    plt.title("Client-wise Accuracy (Fairness View)")
    plt.grid(axis="y")

    gap = max(accs) - min(accs)
    plt.suptitle(f"Fairness Gap (max–min): {gap:.3f}", fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(FED_DIR2, "client_accuracy_bar.png"))
    plt.close()
    print("Saved: client_accuracy_bar.png")


# -------------------------------------------------
# Main
# -------------------------------------------------
def main():
    files = os.listdir(FED_DIR2)

    cm_files = sorted(f for f in files if f.startswith("cm_client_"))
    grad_files = sorted(f for f in files if f.startswith("grad_norms_client_"))
    fairness_json = os.path.join(FED_DIR2, "fairness_metrics.json")

    print(f"Found {len(cm_files)} confusion matrices")
    print(f"Found {len(grad_files)} gradient norm files")

    # Confusion matrices
    for f in cm_files:
        cid = f.split("_")[-1].replace(".npy", "")
        cm = np.load(os.path.join(FED_DIR2, f))
        plot_confusion_matrix(
            cm,
            title=f"Client {cid} – Confusion Matrix",
            out_path=os.path.join(FED_DIR2, f"cm_client_{cid}.png"),
        )

    # Overlay gradient norms
    plot_grad_norms_overlay(grad_files)

    # Fairness metrics
    if os.path.exists(fairness_json):
        plot_fairness_metrics(fairness_json)

    # Client-wise accuracy
    plot_client_accuracy_from_cm(cm_files)


if __name__ == "__main__":
    main()
