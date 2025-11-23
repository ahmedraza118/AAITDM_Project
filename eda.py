# eda.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from data_loader import load_raw_wisdm
from feature_engineering import create_windows_and_features
from config import EDA_DIR, WINDOW_SIZE
import os



def eda_raw_summary(df: pd.DataFrame):
    print("\n=== RAW DATA SUMMARY ===")
    print(df.head())
    print("\nShape:", df.shape)
    print("\nUsers:", df['user'].nunique())
    print("\nActivities distribution:\n", df['activity'].value_counts())

def plot_activity_distribution(df: pd.DataFrame):
    plt.figure(figsize=(8,4))
    sns.countplot(data=df, y="activity", order=df['activity'].value_counts().index)
    plt.title("Raw Sample Distribution per Activity")
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_DIR, "raw_activity_distribution.png"))
    plt.close()

def plot_signal_examples(df: pd.DataFrame):
    #  1 user and 1 activity
    sample_user = df['user'].iloc[1]
    dfu = df[df['user']==sample_user].head(1000)

    plt.figure(figsize=(12,4))
    plt.plot(dfu['timestamp'], dfu['x'], label='x')
    plt.plot(dfu['timestamp'], dfu['y'], label='y')
    plt.plot(dfu['timestamp'], dfu['z'], label='z')
    plt.title(f"Example Raw Accelerometer Signal (User {sample_user})")
    plt.xlabel("Timestamp")
    plt.ylabel("Acceleration")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_DIR, "raw_signal_example2.png"))
    plt.close()


def eda_window_level(sequences, feats, labels):
    print("\n=== WINDOW SUMMARY ===")
    print("Number of windows:", len(sequences))
    print("Window length (samples):", sequences[0].shape[0])

    # engineered feature correlations
    feat_df = pd.DataFrame(feats)
    corr = feat_df.corr()
    plt.figure(figsize=(10,8))
    sns.heatmap(corr, cmap="coolwarm", center=0)
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_DIR, "feature_correlation_heatmap.png"))
    plt.close()

    # Class distribution onwindow level
    plt.figure(figsize=(8,4))
    sns.countplot(y=labels, order=pd.Series(labels).value_counts().index)
    plt.title("Window-Level Activity Distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_DIR, "window_activity_distribution.png"))
    plt.close()


def run_eda():
    df = load_raw_wisdm()

    # RAW EDA
    eda_raw_summary(df)
    plot_activity_distribution(df)
    plot_signal_examples(df)

    # WINDOW EDA
    sequences, feats, labels, users = create_windows_and_features(df)
    eda_window_level(sequences, feats, labels)

    print("\nEDA complete. Plots saved to:", EDA_DIR)

if __name__ == "__main__":
    run_eda()
