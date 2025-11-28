import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from .config import SEED, RESULTS_DIR

def ensure_dirs():
    os.makedirs(RESULTS_DIR, exist_ok=True)

def set_seed():
    np.random.seed(SEED)

def stratified_split(df, test_size=0.2):
    return train_test_split(
        df["review"], df["sentiment"],
        test_size=test_size,
        random_state=SEED,
        stratify=df["sentiment"]
    )

def save_metrics(metrics, filename="metrics.json"):
    ensure_dirs()
    path = os.path.join(RESULTS_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

def save_metrics_csv(rows, filename="metrics.csv"):
    ensure_dirs()
    path = os.path.join(RESULTS_DIR, filename)
    pd.DataFrame(rows).to_csv(path, index=False)

def save_confusion_matrix(cm, filename="confusion_matrix_svm.csv"):
    ensure_dirs()
    path = os.path.join(RESULTS_DIR, filename)
    pd.DataFrame(cm).to_csv(path, index=False)