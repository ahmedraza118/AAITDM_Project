import numpy as np
from sklearn.model_selection import train_test_split
import joblib
import os

def partition_by_user(users, n_clients=5, shuffle=True, random_state=42):
    """
    Given a list/array of user ids (can be strings), partition into n_clients buckets.
    Returns list of lists of users per client.
    """
    unique_users = np.unique(users)
    if shuffle:
        rng = np.random.RandomState(random_state)
        rng.shuffle(unique_users)
    buckets = np.array_split(unique_users, n_clients)
    return [list(b) for b in buckets]
