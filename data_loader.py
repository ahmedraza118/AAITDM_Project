import pandas as pd
import numpy as np
from typing import Tuple
from config import RAW_DATA_PATH

def load_raw_wisdm(raw_path: str = RAW_DATA_PATH) -> pd.DataFrame:
    """
    1679,Walking,1370520469556,0.2941316,-0.6356053,-0.22693644;
    """
    rows = []
    with open(raw_path, "r") as f:
        for ln in f:
            ln = ln.strip().rstrip(";")
            if not ln:
                continue
            parts = ln.split(",")
            if len(parts) < 6:
                continue
            user = parts[0]
            activity = parts[1]
            timestamp = int(parts[2])
            try:
                x = float(parts[3])
                y = float(parts[4])
                z = float(parts[5])
            except ValueError:
                continue
            rows.append((user, activity, timestamp, x, y, z))
    df = pd.DataFrame(rows, columns=["user", "activity", "timestamp", "x", "y", "z"])
    # sort by user, timestamp
    df = df.sort_values(["user", "timestamp"]).reset_index(drop=True)
    return df


if __name__ == "__main__":
    df = load_raw_wisdm()
    print(f"Loaded {len(df)} rows from WISDM raw data.")
    print(df.head())
    print("Data types:")
    print(df.dtypes)
    print("Unique activities:", df['activity'].unique())
    print("Unique users:", df['user'].unique())
