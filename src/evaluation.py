from ensurepip import bootstrap
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from scipy.stats import friedmanchisquare
from classifiers import BayesianKNNClassifier
from sklearn.linear_model import LogisticRegression
from scipy.stats import t

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
    param_grid = {
        "n_neighbors": [1, 3, 5, 7, 9],
        "metric": ["euclidean", "manhattan", "chebyshev"]
    }

    model = BayesianKNNClassifier()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    grid = GridSearchCV(model, param_grid, cv=skf, scoring="f1", error_score='raise')
    grid.fit(X_train, y_train)

    return grid.best_params_


def select_best_logistic(X_train, y_train):
    param_grid = {
        "C": [0.01, 0.1, 1, 10, 100],
        "penalty": ['l1', 'l2'],
        "solver": ['liblinear', 'saga']
    }

    model = LogisticRegression(max_iter=1000)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    grid = GridSearchCV(model, param_grid, cv=skf, scoring="f1", error_score='raise')
    grid.fit(X_train, y_train)

    return grid.best_params_


def intervalo_confianca(data, confidence=0.95):
    data = np.array(data)
    mean = np.mean(data)
    sem = np.std(data, ddof=1) / np.sqrt(len(data))  # erro padrão
    h = sem * t.ppf((1 + confidence) / 2, len(data) - 1)
    return mean, h