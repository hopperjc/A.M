import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from scipy.stats import friedmanchisquare


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
