import os
import pickle
import re

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MultiLabelBinarizer

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(ROOT_DIR, "data", "medical_dataset_clean.csv")
MODELS_DIR = os.path.join(ROOT_DIR, "models")


def clean_and_split(text):
    """Lowercase, strip punctuation (keep commas) and split into symptom tokens."""
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z,\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    parts = [p.strip() for p in text.split(",")]
    return [p for p in parts if len(p) > 2]


def load_features(path=DATA_PATH):
    df = pd.read_csv(path)
    df["Symptoms"] = df["Symptoms"].apply(clean_and_split)
    df = df[df["Symptoms"].map(len) > 0]

    mlb = MultiLabelBinarizer()
    X = mlb.fit_transform(df["Symptoms"])
    y = df["Disease"]
    return X, y, mlb


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)

    X, y, mlb = load_features()

    model = LogisticRegression(max_iter=3000, solver="lbfgs")
    model.fit(X, y)

    with open(os.path.join(MODELS_DIR, "model.pkl"), "wb") as f:
        pickle.dump(model, f)
    with open(os.path.join(MODELS_DIR, "mlb.pkl"), "wb") as f:
        pickle.dump(mlb, f)

    print(f"Training done. Features: {len(mlb.classes_)}, classes: {list(model.classes_)}")


if __name__ == "__main__":
    main()
