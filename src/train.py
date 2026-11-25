"""Train the KNN classifier: baseline, choice of k with cross-validation, GridSearch."""
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedKFold
from .config import CV_FOLDS, RANDOM_STATE


def baseline_knn(X_train, y_train, k=5):
    return KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train)


def k_sweep(X_train, y_train, k_values=range(1, 26, 2)):
    """Mean k-fold CV accuracy for each k (odd values avoid voting ties)."""
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scores = []
    for k in k_values:
        model = KNeighborsClassifier(n_neighbors=k)
        scores.append(cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy").mean())
    return list(k_values), np.array(scores)


def tune_knn(X_train, y_train):
    grid = {
        "n_neighbors": [3, 5, 7, 9, 11, 15],
        "weights": ["uniform", "distance"],
        "metric": ["euclidean", "manhattan"],
    }
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    search = GridSearchCV(KNeighborsClassifier(), grid, cv=cv, scoring="f1", n_jobs=-1)
    search.fit(X_train, y_train)
    return search
