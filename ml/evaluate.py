import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

from ml.train_model import load_features

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGURE_PATH = os.path.join(ROOT_DIR, "docs", "confusion_matrix.png")


def main():
    X, y, _ = load_features()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = LogisticRegression(max_iter=3000, solver="lbfgs")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("===== MODEL EVALUATION (held-out 20%) =====")
    print("Accuracy :", round(accuracy_score(y_test, y_pred), 4))
    print("Precision:", round(precision_score(y_test, y_pred, average="weighted"), 4))
    print("Recall   :", round(recall_score(y_test, y_pred, average="weighted"), 4))
    print("F1-score :", round(f1_score(y_test, y_pred, average="weighted"), 4))
    print()
    print(classification_report(y_test, y_pred))

    labels = sorted(y.unique())
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title("Confusion matrix (held-out test set)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(FIGURE_PATH, dpi=150)
    print(f"Confusion matrix saved to {FIGURE_PATH}")


if __name__ == "__main__":
    main()
