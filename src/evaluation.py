import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from scipy.stats import friedmanchisquare
from classifiers import BayesianKNNClassifier
from sklearn.linear_model import LogisticRegression


def cross_validate_model(model, X, y, n_runs=30, n_folds=10, seed=42):
    rng = np.random.RandomState(seed)
    results = []

    for run in range(n_runs):
        kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=rng.randint(0, 10000))
        for train_idx, test_idx in kf.split(X, y):
            model.fit(X[train_idx], y[train_idx])
            y_pred = model.predict(X[test_idx])
            results.append({
                "accuracy": accuracy_score(y[test_idx], y_pred),
                "precision": precision_score(y[test_idx], y_pred, zero_division=0),
                "recall": recall_score(y[test_idx], y_pred, zero_division=0),
                "f1": f1_score(y[test_idx], y_pred, zero_division=0)
            })

    return pd.DataFrame(results)


def friedman_test(*args):
    """Recebe listas com resultados dos classificadores e aplica o teste de Friedman."""
    return friedmanchisquare(*args)


def summarize_results(df):
    """Calcula média e desvio padrão das métricas."""
    return df.agg(['mean', 'std'])


def prepare_friedman_input(list_of_dfs, metric="accuracy"):
    """Extrai a métrica desejada de múltiplos DataFrames e empacota para Friedman."""
    return [df[metric].values for df in list_of_dfs]


def select_best_bayesian_knn(X_train, y_train):
    best_score = -1
    best_params = {}
    k_values = [1, 3, 5, 7, 9]
    metrics = ["euclidean", "manhattan", "chebyshev"]

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for metric in metrics:
        for k in k_values:
            scores = []
            for train_idx, val_idx in skf.split(X_train, y_train):
                model = BayesianKNNClassifier(n_neighbors=k, metric=metric)
                model.fit(X_train[train_idx], y_train[train_idx])
                y_pred = model.predict(X_train[val_idx])
                scores.append(f1_score(y_train[val_idx], y_pred, zero_division=0))
            avg_score = np.mean(scores)
            if avg_score > best_score:
                best_score = avg_score
                best_params = {"n_neighbors": k, "metric": metric}

    return best_params


def select_best_logistic(X_train, y_train):
    best_score = -1
    best_params = {}
    Cs = [0.01, 0.1, 1, 10, 100]
    penalties = ['l2']  

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for C in Cs:
        for penalty in penalties:
            scores = []
            for train_idx, val_idx in skf.split(X_train, y_train):
                try:
                    model = LogisticRegression(C=C, penalty=penalty, solver='liblinear')
                    model.fit(X_train[train_idx], y_train[train_idx])
                    y_pred = model.predict(X_train[val_idx])
                    scores.append(f1_score(y_train[val_idx], y_pred, zero_division=0))
                except Exception:
                    continue
            avg_score = np.mean(scores)
            if avg_score > best_score:
                best_score = avg_score
                best_params = {"C": C, "penalty": penalty}

    return best_params

