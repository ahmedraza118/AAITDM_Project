import torch
from torch.utils.data import Dataset
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os

class SequenceFeatureDataset(Dataset):
    """
    Each item:
      - seq: (T, 3) float32
      - feat: (F,) float32 (engineered features)
      - label: int
    """
    def __init__(self, sequences, feat_dicts, labels, label_encoder, scaler_feat=None, fit_scaler=False):
        self.sequences = [torch.from_numpy(s) for s in sequences]
        # convert feat_dicts (list of dicts) into array with stable ordering
        if len(feat_dicts) == 0:
            raise ValueError("Empty feature dicts")
        ordered_keys = sorted(feat_dicts[0].keys())
        feat_arr = np.array([[fd[k] for k in ordered_keys] for fd in feat_dicts], dtype=np.float32)
        if scaler_feat is None:
            self.scaler = StandardScaler()
            if fit_scaler:
                self.scaler.fit(feat_arr)
            self.feats = self.scaler.transform(feat_arr) if fit_scaler else feat_arr
        else:
            self.scaler = scaler_feat
            self.feats = self.scaler.transform(feat_arr)
        self.labels = labels
        self.label_encoder = label_encoder
        self.ordered_keys = ordered_keys

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        seq = self.sequences[idx]        # shape (T,3)
        feat = torch.from_numpy(self.feats[idx])  # (F,)
        label = int(self.label_encoder.transform([self.labels[idx]])[0])
        return seq, feat.float(), torch.tensor(label, dtype=torch.long)

    def save_scaler(self, path):
        joblib.dump(self.scaler, path)

    def get_feature_keys(self):
        return self.ordered_keys

