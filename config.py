# config.py
import os

# Data paths
# ROOT = os.path.dirname(os.path.abspath(__file__))
ROOT = "./"
RAW_DATA_PATH = os.path.join(ROOT, "data/WISDM_at_v2.0_raw.txt") 
DEMOGRAPHICS_PATH = os.path.join(ROOT, "data/WISDM_at_v2.0_demographics.txt")



# Output / artifacts
ARTIFACTS_DIR = os.path.join(ROOT, "artifacts")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

EDA_DIR = os.path.join(ROOT, "eda")
os.makedirs(EDA_DIR, exist_ok=True)

FED_DIR = os.path.join(ARTIFACTS_DIR, "federated")
os.makedirs(FED_DIR, exist_ok=True)

CENTRAL_DIR = os.path.join(ARTIFACTS_DIR, "central")
os.makedirs(CENTRAL_DIR, exist_ok=True)


FED_DIR2 = os.path.join(ARTIFACTS_DIR, "federated3")
os.makedirs(FED_DIR2, exist_ok=True)

# Windowing
SAMPLE_RATE_HZ = 20
WINDOW_SECONDS = 10
WINDOW_SIZE = SAMPLE_RATE_HZ * WINDOW_SECONDS  # 200

# Training params
NUM_CLASSES = 6  # walking, jogging, stairs, sitting, standing, lyingdown
BATCH_SIZE = 64
CENTRAL_EPOCHS = 30
LOCAL_EPOCHS = 3
LEARNING_RATE = 1e-3
NUM_CLIENTS = 5
RANDOM_SEED = 42
# DEVICE = "cuda" if __import__("torch").cuda.is_available() else "cpu"
DEVICE = "cpu"

FED_ROUNDS = 30


# ---- Privacy ----
DP_ENABLED = True
DP_CLIP_NORM = 1.0
DP_NOISE_MULTIPLIER = 0.8  # sigma
DP_DELTA = 1e-5

# ---- Robustness ----
POISONING_ENABLED = True
POISON_CLIENT_FRACTION = 0.3
LABEL_NOISE_RATE = 0.2  # 20% labels flipped




# if __name__ == "__main__":
#     print("Configuration:")
#     print(f"Root Directory: {ROOT}")
#     print(f"Raw Data Path: {RAW_DATA_PATH}")
#     print(f"Demographics Path: {DEMOGRAPHICS_PATH}")
#     print(f"Artifacts Directory: {ARTIFACTS_DIR}")
