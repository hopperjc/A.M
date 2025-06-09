import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
import wandb
from joblib import Parallel, delayed
from imblearn.over_sampling import SMOTE

from sklearn.model_selection import StratifiedKFold, GridSearchCV, learning_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.exceptions import ConvergenceWarning

# Importe sua classe customizada (ajuste o caminho se necessário)
from src.gaussian_bayesian_classifier import GaussianBayesClassifier

from scipy.stats import friedmanchisquare, t
import scikit_posthocs as sp

# --- CONFIGURAÇÃO INICIAL ---
# Suprimir warnings para uma saída mais limpa
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)


# --- 1. FUNÇÕES DE SETUP E AVALIAÇÃO ---

def get_models_and_grids():
    """Retorna dicionários de modelos base e seus grids de hiperparâmetros."""
    knn_pipeline = Pipeline([('scaler', StandardScaler()), ('knn', KNeighborsClassifier())])
    logreg_pipeline = Pipeline([('scaler', StandardScaler()), ('logreg', LogisticRegression(max_iter=5000, random_state=42))])
    base_models = {"KNN": knn_pipeline, "LogisticRegression": logreg_pipeline}
    param_grids = {
        "KNN": {'knn__n_neighbors': np.arange(1, 32, 2), 'knn__metric': ['euclidean', 'manhattan', 'chebyshev']},
        "LogisticRegression": {'logreg__penalty': ['l1', 'l2'], 'logreg__C': np.logspace(-3, 3, 7), 'logreg__solver': ['liblinear', 'saga']}
    }
    return base_models, param_grids

def run_single_repetition(X, y, repetition_seed, n_outer_splits, n_inner_splits):
    """Executa UMA repetição completa de 10-fold CV com SMOTE nos dados de treino."""
    base_models, param_grids = get_models_and_grids()
    classifier_names = ["GaussianBayes", "KNN", "LogisticRegression", "Voting"]
    metrics = {
        "Error Rate": (lambda yt, yp: 1 - accuracy_score(yt, yp)),
        "Precision": (lambda yt, yp: precision_score(yt, yp, zero_division=0)),
        "Recall": (lambda yt, yp: recall_score(yt, yp, zero_division=0)),
        "F1-Score": (lambda yt, yp: f1_score(yt, yp, zero_division=0))
    }
    repetition_results = {model: {metric: [] for metric in metrics} for model in classifier_names}
    outer_cv = StratifiedKFold(n_splits=n_outer_splits, shuffle=True, random_state=repetition_seed)

    for train_idx, test_idx in outer_cv.split(X, y):
        X_train, X_test, y_train, y_test = X[train_idx], X[test_idx], y[train_idx], y[test_idx]

        # --- APLICAÇÃO DO SMOTE (A MUDANÇA PRINCIPAL) ---
        # 1. Instanciamos o SMOTE. Usamos o random_state para reprodutibilidade.
        smote = SMOTE(random_state=repetition_seed)
        # 2. Aplicamos o SMOTE APENAS nos dados de treino deste fold.
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
        # --- FIM DA MUDANÇA ---

        inner_cv = StratifiedKFold(n_splits=n_inner_splits, shuffle=True, random_state=repetition_seed)
        trained_models_for_fold, estimators_for_voting = {}, []

        # --- Modelo Bayesiano Gaussiano ---
        # Treinamos com os dados rebalanceados
        gnb_trained = GaussianBayesClassifier().fit(X_train_res, y_train_res)
        trained_models_for_fold["GaussianBayes"] = gnb_trained
        estimators_for_voting.append(('gnb', GaussianBayesClassifier()))

        # --- Modelos com GridSearchCV ---
        for name, model in base_models.items():
            grid_search = GridSearchCV(estimator=model, param_grid=param_grids[name], cv=inner_cv, scoring='f1', n_jobs=1)
            # O GridSearchCV é ajustado com os dados rebalanceados
            grid_search.fit(X_train_res, y_train_res)
            trained_models_for_fold[name] = grid_search.best_estimator_
            model_for_voting = model.set_params(**grid_search.best_params_)
            estimators_for_voting.append((name.lower(), model_for_voting))

        # --- Voting Classifier ---
        # O VotingClassifier também é treinado com os dados rebalanceados
        voting_clf = VotingClassifier(estimators=estimators_for_voting, voting='hard')
        voting_clf.fit(X_train_res, y_train_res)
        trained_models_for_fold["Voting"] = voting_clf

        # --- Avaliação de todos os modelos ---
        # A avaliação continua sendo feita no conjunto de teste ORIGINAL e INTOCADO
        for model_name, model in trained_models_for_fold.items():
            y_pred = model.predict(X_test)
            for metric_name, metric_func in metrics.items():
                repetition_results[model_name][metric_name].append(metric_func(y_test, y_pred))
    return repetition_results

def run_nested_cv_evaluation(X, y, n_repetitions=30, n_outer_splits=10, n_inner_splits=5):
    """Orquestra a avaliação paralela usando Joblib."""
    print(f"Iniciando avaliação {n_repetitions}x{n_outer_splits}-Fold com Nested CV (em paralelo)...")
    parallel_results = Parallel(n_jobs=-1)(
        delayed(run_single_repetition)(X, y, i, n_outer_splits, n_inner_splits)
        for i in tqdm(range(n_repetitions), desc="Repetições")
    )
    classifier_names = ["GaussianBayes", "KNN", "LogisticRegression", "Voting"]
    metric_names = ["Error Rate", "Precision", "Recall", "F1-Score"]
    aggregated_results = {model: {metric: [] for metric in metric_names} for model in classifier_names}
    for single_run_result in parallel_results:
        for model in classifier_names:
            for metric in metric_names:
                aggregated_results[model][metric].extend(single_run_result[model][metric])
    print("Avaliação concluída.")
    return aggregated_results


# --- 2. FUNÇÕES DE ANÁLISE E VISUALIZAÇÃO ---

def analyze_and_print_results(results):
    """(item b) Calcula e apresenta a tabela de métricas com IC 95%."""
    print("\n--- (b) Análise de Desempenho dos Classificadores ---\n")
    summary_data = []
    n_total_scores = len(next(iter(next(iter(results.values())).values())))
    for model_name, metric_results in results.items():
        for metric_name, scores in metric_results.items():
            mean_score, std_dev = np.mean(scores), np.std(scores)
            t_crit = t.ppf(0.975, df=n_total_scores - 1)
            ci_margin = t_crit * (std_dev / np.sqrt(n_total_scores))
            summary_data.append({
                "Classifier": model_name, "Metric": metric_name, "Mean": f"{mean_score:.4f}",
                "Std Dev": f"{std_dev:.4f}", "95% CI": f"[{mean_score - ci_margin:.4f}, {mean_score + ci_margin:.4f}]"
            })
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string())
    wandb.log({"performance_summary": wandb.Table(dataframe=summary_df)})

def perform_statistical_tests(results):
    """(item c) Executa testes de Friedman e Nemenyi com heatmaps."""
    print("\n--- (c) Testes Estatísticos de Comparação ---\n")
    classifier_names = list(results.keys())
    for metric_name in next(iter(results.values())).keys():
        print(f"--- Teste para a métrica: {metric_name} ---")
        scores_df = pd.DataFrame({model: results[model][metric_name] for model in results})
        try:
            stat, p_value = friedmanchisquare(*[scores_df[col] for col in scores_df.columns])
            print(f"Teste de Friedman: Estatística={stat:.4f}, p-valor={p_value:.4f}")
            wandb.log({f"Friedman/{metric_name}_statistic": stat, f"Friedman/{metric_name}_p_value": p_value})
            if p_value < 0.05:
                print("Diferença significante encontrada. Gerando heatmap do pós-teste de Nemenyi...")
                nemenyi_results = sp.posthoc_nemenyi_friedman(scores_df)
                nemenyi_results.columns = classifier_names
                nemenyi_results.index = classifier_names

                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(nemenyi_results, annot=True, fmt=".4f", cmap="coolwarm_r", ax=ax)
                ax.set_title(f"P-valores do Teste de Nemenyi ({metric_name})")
                plt.tight_layout()
                wandb.log({f"Nemenyi Heatmap/{metric_name}": wandb.Image(fig)})
                plt.show()
            else:
                print("Nenhuma diferença estatisticamente significante detectada.")
        except ValueError as e:
            print(f"Não foi possível realizar o teste de Friedman para a métrica {metric_name}: {e}")
        print("-" * 50 + "\n")

def plot_learning_curves(X, y):
    """(item d) Gera e plota as curvas de aprendizagem."""
    print("\n--- (d) Gerando Curvas de Aprendizagem para o Classificador Bayesiano Gaussiano ---\n")
    estimator = GaussianBayesClassifier()
    train_sizes_abs = np.linspace(0.05, 1.0, 20)
    metrics = {
        "Accuracy": make_scorer(accuracy_score),
        "Precision": make_scorer(precision_score, zero_division=0),
        "Recall": make_scorer(recall_score, zero_division=0),
        "F1-Score": make_scorer(f1_score, zero_division=0)
    }
    for metric_name, scorer in metrics.items():
        train_sizes, train_scores, test_scores = learning_curve(
            estimator, X, y, cv=StratifiedKFold(10), n_jobs=-1, train_sizes=train_sizes_abs, scoring=scorer)
        train_scores_mean, train_scores_std = np.mean(train_scores, axis=1), np.std(train_scores, axis=1)
        test_scores_mean, test_scores_std = np.mean(test_scores, axis=1), np.std(test_scores, axis=1)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_title(f"Curva de Aprendizagem (Gaussiano) - Métrica: {metric_name}")
        ax.set_xlabel("Tamanho do Conjunto de Treinamento"); ax.set_ylabel(metric_name); ax.grid()
        ax.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
        ax.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g")
        ax.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Score de Treinamento")
        ax.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Score de Validação Cruzada")
        ax.legend(loc="best")
        wandb.log({f"Learning Curves/{metric_name}": wandb.Image(fig)})
        plt.show()

# --- ORQUESTRADOR PRINCIPAL ---
if __name__ == "__main__":
    try:
        with open("SPECTF.pkl", "rb") as f:
            X, y = pickle.load(f)
    except FileNotFoundError:
        print("Erro: Arquivo 'SPECTF.pkl' não encontrado. Certifique-se de que está no diretório correto.")
        exit()

    run = wandb.init(project="ML-Classifier-Comparison-30", name="SPECTF-Experiment-Combined")
    wandb.config.n_repetitions = 30
    wandb.config.n_outer_splits = 10
    wandb.config.n_inner_splits = 5

    # Executar avaliação, análise, testes e curvas
    all_results = run_nested_cv_evaluation(X, y, n_repetitions=wandb.config.n_repetitions)
    analyze_and_print_results(all_results)
    perform_statistical_tests(all_results)
    plot_learning_curves(X, y)

    run.finish()
    print("\nExecução finalizada! Resultados salvos no seu dashboard do Weights & Biases.")