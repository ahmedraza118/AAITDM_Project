import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
import matplotlib.pyplot as plt
import seaborn as sns
import os
from config import CENTRAL_DIR



def plot_training_curves(history, save_prefix="central"):
    """
    {
        "loss": [L1, L2, ...],
        "acc":  [A1, A2, ...],
        "f1":   [F1, F2, ...]
    }
    """
    epochs = list(range(1, len(history["loss"]) + 1))

    # ---- Loss ----
    plt.figure(figsize=(7, 5))
    plt.plot(epochs, history["loss"], marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{save_prefix.upper()} - Loss Curve")
    plt.grid(True)
    plt.savefig(os.path.join(CENTRAL_DIR, f"{save_prefix}_loss_curve.png"),
                bbox_inches="tight")
    plt.close()

    # ---- Accuracy ----
    plt.figure(figsize=(7, 5))
    plt.plot(epochs, history["acc"], marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"{save_prefix.upper()} - Accuracy Curve")
    # plt.ylim(0, 1)
    plt.grid(True)
    plt.savefig(os.path.join(CENTRAL_DIR, f"{save_prefix}_accuracy_curve.png"),
                bbox_inches="tight")
    plt.close()

    # ---- F1 Macro ----
    plt.figure(figsize=(7, 5))
    plt.plot(epochs, history["f1"], marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("F1 Macro")
    plt.title(f"{save_prefix.upper()} - F1 Curve")
    # plt.ylim(0, 1)
    plt.grid(True)
    plt.savefig(os.path.join(CENTRAL_DIR, f"{save_prefix}_f1_curve.png"),
                bbox_inches="tight")
    plt.close()

    print("✔ Saved training curves for central model.")



def evaluate_and_report(
    model,
    dataloader,
    device,
    label_encoder=None,
    return_preds=False,
    save_prefix="central",
):
    model.eval()
    ys, ypred = [], []

    with __import__("torch").no_grad():
        for seq, feat, label in dataloader:
            seq = seq.to(device)
            feat = feat.to(device)

            logits = model(seq, feat)
            preds = logits.argmax(dim=1).cpu().numpy()

            ys.append(label.numpy())
            ypred.append(preds)

    ys = np.concatenate(ys)
    ypred = np.concatenate(ypred)

    # ---- Metrics ----
    acc = accuracy_score(ys, ypred)
    f1 = f1_score(ys, ypred, average="macro")
    cm = confusion_matrix(ys, ypred)
    cm_norm = confusion_matrix(ys, ypred, normalize="true")

    report = classification_report(ys, ypred, output_dict=True)

    # ---- Labels ----
    if label_encoder is not None:
        labels = label_encoder.classes_
    else:
        labels = [str(i) for i in range(cm.shape[0])]


    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=labels,
        yticklabels=labels,
        cmap="Blues",
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"{save_prefix.upper()} - Confusion Matrix (Raw)")
    plt.savefig(
        os.path.join(CENTRAL_DIR, f"{save_prefix}_confusion_matrix_raw.png"),
        bbox_inches="tight",
    )
    plt.close()

    # NORMALIZED CONFUSION MATRIX
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        xticklabels=labels,
        yticklabels=labels,
        cmap="Blues",
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"{save_prefix.upper()} - Confusion Matrix (Normalized)")
    plt.savefig(
        os.path.join(CENTRAL_DIR, f"{save_prefix}_confusion_matrix_normalized.png"),
        bbox_inches="tight",
    )
    plt.close()

    # save in text file
    with open(os.path.join(CENTRAL_DIR, f"{save_prefix}_metrics.txt"), "w") as f:
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"F1 Macro: {f1:.4f}\n")
        f.write("\nClassification Report:\n")
        f.write(classification_report(ys, ypred))

    results = {
        "acc": acc,
        "f1_macro": f1,
        "cm": cm,
        "cm_norm": cm_norm,
        "report": report,
    }

    if return_preds:
        results["y_true"] = ys
        results["y_pred"] = ypred

    return results
