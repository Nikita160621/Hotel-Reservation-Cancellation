"""Run the full pipeline:  python main.py
Outputs: figures in reports/figures/, metrics in reports/results.txt, model in models/."""
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from src.config import MODEL_DIR, RESULTS_PATH
from src.data_loader import load_data, quality_report
from src.eda import run_all_eda
from src.preprocess import prepare
from src.train import baseline_knn, k_sweep, tune_knn
from src.evaluate import (compute_metrics, text_report, plot_confusion, plot_k_sweep,
                          plot_roc, plot_model_comparison)


def main():
    raw = load_data()
    lines = ["=== DATA QUALITY ===", str(quality_report(raw))]

    run_all_eda(raw)

    X_train, X_test, y_train, y_test, scaler, feature_names = prepare(raw)
    lines.append(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    # 1. Baseline KNN (k=5)
    base = baseline_knn(X_train, y_train, k=5)
    base_metrics = compute_metrics(base, X_test, y_test)

    # 2. Choose k by cross-validation
    k_values, scores = k_sweep(X_train, y_train)
    plot_k_sweep(k_values, scores)
    lines.append(f"Best k by CV accuracy: {k_values[scores.argmax()]}")

    # 3. Tuned KNN with GridSearchCV
    search = tune_knn(X_train, y_train)
    best = search.best_estimator_
    lines.append(f"Best params: {search.best_params_}")
    tuned_metrics = compute_metrics(best, X_test, y_test)

    # 4. Comparison model (Day 3: Logistic Regression)
    logreg = LogisticRegression(max_iter=1000).fit(X_train, y_train)
    log_metrics = compute_metrics(logreg, X_test, y_test)

    results = {"KNN (k=5)": base_metrics, "KNN (tuned)": tuned_metrics, "Logistic Regression": log_metrics}
    for name, m in results.items():
        lines.append(f"{name}: " + ", ".join(f"{k}={v:.4f}" for k, v in m.items()))
    lines += ["", "=== Tuned KNN classification report ===", text_report(best, X_test, y_test)]

    plot_confusion(best, X_test, y_test)
    plot_roc(best, X_test, y_test)
    plot_model_comparison(results)

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump({"model": best, "scaler": scaler, "features": feature_names}, MODEL_DIR / "knn_model.joblib")

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
