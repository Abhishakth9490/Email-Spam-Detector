"""
Load the trained spam classifier and run predictions on new email
feature vectors (58 Spambase-style columns, no 'class' column).

Usage:
    python3 src/predict.py path/to/new_emails.csv
"""

import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "outputs" / "model"


def load_pipeline():
    model = joblib.load(MODEL_DIR / "best_model.pkl")
    scaler = joblib.load(MODEL_DIR / "scaler.pkl")
    return model, scaler


def predict(csv_path: str):
    model, scaler = load_pipeline()
    X_new = pd.read_csv(csv_path)
    X_scaled = scaler.transform(X_new)
    preds = model.predict(X_scaled)
    proba = model.predict_proba(X_scaled)[:, 1]

    out = X_new.copy()
    out["predicted_class"] = preds
    out["spam_probability"] = proba.round(4)
    return out


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 src/predict.py path/to/new_emails.csv")
        sys.exit(1)

    result = predict(sys.argv[1])
    print(result[["predicted_class", "spam_probability"]])
