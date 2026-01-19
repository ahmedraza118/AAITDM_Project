import os
import joblib
from data_loader import load_raw_wisdm
from feature_engineering import create_windows_and_features
from prepare_data import SequenceFeatureDataset
from train_central import train, prepare_data_and_dataset
from evaluate import evaluate_and_report
# from federated import run_federated_simulation
# from federated_privacy import run_federated_simulation
from federated_xai import run_federated_simulation

from utils import partition_by_user
from config import FED_DIR2, NUM_CLIENTS
from sklearn.preprocessing import LabelEncoder
import numpy as np
from torch.utils.data import DataLoader

def prepare_for_federation(raw_path=None, n_clients=NUM_CLIENTS):
    df = load_raw_wisdm(raw_path) if raw_path else load_raw_wisdm()
    sequences, feats, labels, users = create_windows_and_features(df)
    le = LabelEncoder()
    le.fit(labels)
    joblib.dump(le, os.path.join(FED_DIR2, "label_encoder.joblib"))
    # Partition users into clients
    buckets = partition_by_user(users, n_clients)
    client_datasets = []
    
    # Fit a global scaler on all features
    ordered_keys = sorted(feats[0].keys())
    feat_arr = np.array([[fd[k] for k in ordered_keys] for fd in feats], dtype=float)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(feat_arr)
    joblib.dump(scaler, os.path.join(FED_DIR2, "feature_scaler.joblib"))
    # allocate windows to clients by owner user id
    for client_users in buckets:
        idxs = [i for i,u in enumerate(users) if u in client_users]
        seq_c = [sequences[i] for i in idxs]
        feat_c = [feats[i] for i in idxs]
        lab_c = [labels[i] for i in idxs]
        if len(lab_c) == 0:
            # avoid empty client
            continue
        ds = SequenceFeatureDataset(seq_c, feat_c, lab_c, le, scaler_feat=scaler, fit_scaler=False)
        client_datasets.append(ds)
    print(f"Prepared {len(client_datasets)} client datasets.")
    return client_datasets, le

if __name__ == "__main__":
    # 1) Centralized baseline
    # print("Preparing centralized data and training baseline...")
    # ds_train, ds_val, label_encoder = prepare_data_and_dataset()
    # model = train(ds_train, ds_val, num_classes=len(label_encoder.classes_))

    # 2) Prepare federated client datasets (non-IID by user)
    print("Preparing federated datasets...")
    client_datasets, le = prepare_for_federation()
    feat_dim = len(client_datasets[0].get_feature_keys())
    num_classes = len(le.classes_)
    print("Starting federated simulation...")
    run_federated_simulation(client_datasets, feat_dim, num_classes)
    print("Federated simulation finished.")
    # from interpretability import run_interpretability
    # # After federated training
    # run_interpretability(client_datasets, feat_dim, num_classes)

