# train_central.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
import os
import joblib

from config import BATCH_SIZE, CENTRAL_EPOCHS, LEARNING_RATE, DEVICE, CENTRAL_DIR
from model import ConvLSTMClassifier
from prepare_data import SequenceFeatureDataset
from feature_engineering import create_windows_and_features
from data_loader import load_raw_wisdm
from sklearn.preprocessing import LabelEncoder

from evaluate import evaluate_and_report, plot_training_curves






def prepare_data_and_dataset(raw_path=None, test_size=0.2, random_state=42):
    df = load_raw_wisdm(raw_path) if raw_path else load_raw_wisdm()
    sequences, feats, labels, users = create_windows_and_features(df)

    # label encoder fit
    le = LabelEncoder()
    le.fit(labels)
    joblib.dump(le, os.path.join(CENTRAL_DIR, "label_encoder.joblib"))

    idx = np.arange(len(labels))
    X_train_idx, X_val_idx = train_test_split(
        idx, test_size=test_size, stratify=labels, random_state=random_state
    )

    seq_train = [sequences[i] for i in X_train_idx]
    feat_train = [feats[i] for i in X_train_idx]
    lab_train = [labels[i] for i in X_train_idx]

    seq_val = [sequences[i] for i in X_val_idx]
    feat_val = [feats[i] for i in X_val_idx]
    lab_val = [labels[i] for i in X_val_idx]

    # fit scaler on train features via dataset
    ds_train = SequenceFeatureDataset(
        seq_train, feat_train, lab_train, le, scaler_feat=None, fit_scaler=True
    )
    ds_train.save_scaler(os.path.join(CENTRAL_DIR, "feature_scaler.joblib"))

    ds_val = SequenceFeatureDataset(
        seq_val, feat_val, lab_val, le, scaler_feat=ds_train.scaler, fit_scaler=False
    )

    return ds_train, ds_val, le



def train(ds_train, ds_val, num_classes=6):
    train_loader = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(ds_val, batch_size=BATCH_SIZE, shuffle=False)

    feat_dim = len(ds_train.get_feature_keys())
    model = ConvLSTMClassifier(
        seq_channels=3, feat_dim=feat_dim, num_classes=num_classes
    ).to(DEVICE)

    opt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    best_val = 0.0

    history = {"loss": [], "acc": [], "f1": []}

    for ep in range(1, CENTRAL_EPOCHS + 1):

        model.train()
        total_loss = 0.0

        for seq, feat, label in train_loader:
            seq = seq.to(DEVICE)
            feat = feat.to(DEVICE)
            label = label.to(DEVICE)

            opt.zero_grad()
            logits = model(seq, feat)
            loss = criterion(logits, label)
            loss.backward()
            opt.step()

            total_loss += loss.item() * seq.size(0)

        avg_loss = total_loss / len(train_loader.dataset)

        # validate
        metrics = evaluate_and_report(
            model, val_loader, DEVICE, return_preds=False, save_prefix="central_epoch"
        )

        val_acc = metrics["acc"]
        val_f1 = metrics["f1_macro"]

        history["loss"].append(avg_loss)
        history["acc"].append(val_acc)
        history["f1"].append(val_f1)

        print(
            f"[Central] Epoch {ep}/{CENTRAL_EPOCHS} "
            f"loss={avg_loss:.4f} val_acc={val_acc:.4f} val_f1={val_f1:.4f}"
        )

        # Save model
        if val_acc > best_val:
            best_val = val_acc
            torch.save(
                model.state_dict(), os.path.join(CENTRAL_DIR, "best_central_model.pt")
            )

    print("Finished centralized training. Best val acc:", best_val)

    plot_training_curves(history, save_prefix="central")

    return model



if __name__ == "__main__":
    ds_train, ds_val, le = prepare_data_and_dataset()
    model = train(ds_train, ds_val, num_classes=len(le.classes_))
    print("=" * 40)
    print("Centralized training complete.")
