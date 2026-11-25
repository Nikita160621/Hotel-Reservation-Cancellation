"""Metrics and evaluation plots (Day 2: accuracy, precision, recall, confusion matrix)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, classification_report, roc_curve, roc_auc_score)
from .config import FIG_DIR


def _save(name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(FIG_DIR / name, dpi=150)
    plt.close()


def compute_metrics(model, X_test, y_test) -> dict:
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred),
        "recall": recall_score(y_test, pred),
        "f1": f1_score(y_test, pred),
        "roc_auc": roc_auc_score(y_test, proba),
    }


def text_report(model, X_test, y_test) -> str:
    return classification_report(y_test, model.predict(X_test),
                                 target_names=["Not_Canceled", "Canceled"])


def plot_confusion(model, X_test, y_test, filename="10_confusion_matrix.png", title="Confusion matrix (test set)"):
    cm = confusion_matrix(y_test, model.predict(X_test))
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Not_Canceled", "Canceled"], yticklabels=["Not_Canceled", "Canceled"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    _save(filename)


def plot_k_sweep(k_values, scores):
    plt.figure(figsize=(7, 4))
    plt.plot(k_values, scores, marker="o")
    best = k_values[scores.argmax()]
    plt.axvline(best, linestyle="--", color="grey")
    plt.xlabel("k (number of neighbours)")
    plt.ylabel("Mean 5-fold CV accuracy")
    plt.title(f"Choosing k (best k = {best})")
    _save("09_k_vs_accuracy.png")


def plot_roc(model, X_test, y_test):
    proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, proba)
    auc = roc_auc_score(y_test, proba)
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label=f"KNN (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], "--", color="grey")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC curve")
    plt.legend()
    _save("11_roc_curve.png")


def plot_model_comparison(results: dict):
    """results: {model_name: metrics_dict}"""
    names = list(results)
    metrics = ["accuracy", "precision", "recall", "f1"]
    width = 0.2
    plt.figure(figsize=(9, 4))
    for i, m in enumerate(metrics):
        plt.bar([x + i * width for x in range(len(names))], [results[n][m] for n in names], width, label=m)
    plt.xticks([x + 1.5 * width for x in range(len(names))], names, rotation=15)
    plt.ylim(0, 1)
    plt.title("Model comparison (test set)")
    plt.legend(loc="lower right")
    _save("12_model_comparison.png")
