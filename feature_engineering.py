import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from typing import List, Tuple, Dict
from config import WINDOW_SIZE

def sliding_windows_for_user(df_user: pd.DataFrame, window_size: int = WINDOW_SIZE, step: int = WINDOW_SIZE) -> List[pd.DataFrame]:
    """
    Cut user's time-series into non-overlapping windows of fixed number of samples.
    Each window is a DataFrame with window_size rows.
    """
    windows = []
    n = len(df_user)
    i = 0
    while i + window_size <= n:
        win = df_user.iloc[i:i+window_size]
        windows.append(win)
        i += step
    return windows

def dominant_frequency(signal: np.ndarray, fs: int = 20) -> float:
    # using rfft to find dominant frequency (Hz)
    sig = signal - np.mean(signal)
    N = len(sig)
    if N == 0:
        return 0.0
    freqs = np.fft.rfftfreq(N, d=1.0/fs)
    fft = np.abs(np.fft.rfft(sig))
    if fft.size == 0:
        return 0.0
    dominant = freqs[np.argmax(fft)]
    return float(dominant)

def compute_bins(axis_vals: np.ndarray, num_bins: int = 10, vmin: float = -20.0, vmax: float = 20.0) -> List[float]:
    # fraction of values in each bin
    bins = np.linspace(vmin, vmax, num_bins+1)
    counts, _ = np.histogram(axis_vals, bins=bins)
    frac = counts / len(axis_vals)
    return frac.tolist()

def compute_features_for_window(win_df: pd.DataFrame) -> Dict:
    """
    Compute engineered features for a window (DataFrame with x,y,z columns).
    Compute similar  features described in the transformed.arff.
    """
    x = win_df['x'].values
    y = win_df['y'].values
    z = win_df['z'].values
    features = {}
    # bins X0..X9
    xb = compute_bins(x)
    yb = compute_bins(y)
    zb = compute_bins(z)
    for i, v in enumerate(xb):
        features[f"X{i}"] = v
    for i, v in enumerate(yb):
        features[f"Y{i}"] = v
    for i, v in enumerate(zb):
        features[f"Z{i}"] = v
    # averages
    features["XAVG"] = float(np.mean(x))
    features["YAVG"] = float(np.mean(y))
    features["ZAVG"] = float(np.mean(z))
    # stdev
    features["XSTD"] = float(np.std(x))
    features["YSTD"] = float(np.std(y))
    features["ZSTD"] = float(np.std(z))
    # abs dev
    features["XABSDEV"] = float(np.mean(np.abs(x - np.mean(x))))
    features["YABSDEV"] = float(np.mean(np.abs(y - np.mean(y))))
    features["ZABSDEV"] = float(np.mean(np.abs(z - np.mean(z))))
    # resultant mean
    resultant = np.sqrt(x**2 + y**2 + z**2)
    features["RESULTANT"] = float(np.mean(resultant))
    # dominant freq approximations
    features["XPEAK"] = dominant_frequency(x)
    features["YPEAK"] = dominant_frequency(y)
    features["ZPEAK"] = dominant_frequency(z)
    # simple peak count
    features["XPEAKCOUNT"] = int(len(find_peaks(x, distance=5)[0]))
    features["YPEAKCOUNT"] = int(len(find_peaks(y, distance=5)[0]))
    features["ZPEAKCOUNT"] = int(len(find_peaks(z, distance=5)[0]))
    return features

def create_windows_and_features(df: pd.DataFrame, window_size: int = WINDOW_SIZE, step: int = WINDOW_SIZE) -> Tuple[List[np.ndarray], List[Dict], List[str], List[str]]:
    """
    Returns:
      sequences: list of shape (window_size, 3) numpy arrays
      feat_dicts: list of dicts for engineered features
      labels: list of activity labels (string)
      users: list of user ids (string)
    """
    sequences = []
    feats = []
    labels = []
    users = []
    for user, df_user in df.groupby("user"):
        df_user = df_user.sort_values("timestamp")
        wins = sliding_windows_for_user(df_user, window_size=window_size, step=step)
        for win in wins:
            # majority label in window; if mixed labels pick majority
            activity = win['activity'].mode().iloc[0] if not win['activity'].mode().empty else "NoLabel"
            if activity == "NoLabel":
                continue
            seq = win[['x','y','z']].values.astype(np.float32)
            feat = compute_features_for_window(win)
            sequences.append(seq)
            feats.append(feat)
            labels.append(activity)
            users.append(user)
    return sequences, feats, labels, users


if __name__ == "__main__":
    from data_loader import load_raw_wisdm
    df = load_raw_wisdm()
    sequences, feats, labels, users = create_windows_and_features(df)
    print(f"Created {len(sequences)} windows with features.")
    print("Example features from first window:")
    print(feats[0] if feats else "No features computed.")