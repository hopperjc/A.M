import numpy as np
from scipy.linalg import LinAlgError
import warnings
from sklearn.base import BaseEstimator, ClassifierMixin
# O import de 'validate_data' foi adicionado aqui
from sklearn.utils.validation import check_is_fitted, validate_data
from sklearn.utils.multiclass import type_of_target
from scipy.special import logsumexp

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report


# --- IMPLEMENTAÇÃO CLASSIFICADOR BAYESIANO GAUSSIANO ---
class GaussianBayesClassifier(ClassifierMixin, BaseEstimator):
    """ Classificador Bayesiano Gaussiano com MLE para parâmetros. """
    _estimator_type = "classifier"

    def __init__(self, reg_epsilon=1e-6):
        self.reg_epsilon = reg_epsilon

    def fit(self, X, y):
        X, y = validate_data(self, X, y)

        y_type = type_of_target(y)
        if y_type == "continuous":
            raise ValueError("Unknown label type: 'continuous' is not supported.")

        self.classes_ = np.unique(y)
        self.priors_ = {}
        self.means_ = {}
        self.inv_covariances_ = {}
        self.log_det_covariances_ = {}

        for c in self.classes_:
            X_c = X[y == c]
            n_samples_c = X_c.shape[0]
            if n_samples_c == 0: continue
            self.priors_[c] = n_samples_c / X.shape[0]
            self.means_[c] = np.mean(X_c, axis=0)
            if n_samples_c <= self.n_features_in_:
                warnings.warn(f"Classe {c} tem poucas amostras.")
            cov = np.cov(X_c, rowvar=False)
            cov_reg = cov + np.eye(self.n_features_in_) * self.reg_epsilon
            try:
                self.inv_covariances_[c] = np.linalg.inv(cov_reg)
                sign, logdet = np.linalg.slogdet(cov_reg)
                if sign <= 0: raise LinAlgError("Log-determinante não é positivo.")
                self.log_det_covariances_[c] = logdet
            except LinAlgError as e:
                raise LinAlgError(f"Erro na classe {c}: Matriz de covariância singular. {e}")
        return self

    def _log_likelihood(self, X):
        check_is_fitted(self)
        log_likelihoods = np.zeros((X.shape[0], len(self.classes_)))
        for i, c in enumerate(self.classes_):
            mean, inv_cov, log_det_cov = self.means_[c], self.inv_covariances_[c], self.log_det_covariances_[c]
            log_pdf_constant = -0.5 * (self.n_features_in_ * np.log(2 * np.pi) + log_det_cov)
            diff = X - mean
            exponent_term = -0.5 * np.sum(np.dot(diff, inv_cov) * diff, axis=1)
            log_likelihoods[:, i] = log_pdf_constant + exponent_term
        return log_likelihoods

    def _log_posterior(self, X):
        log_likelihood = self._log_likelihood(X)
        log_priors = np.log([self.priors_.get(c, 1e-9) for c in self.classes_])
        return log_likelihood + log_priors

    def predict_log_proba(self, X):
        X = validate_data(self, X, reset=False)
        log_posterior_unnormalized = self._log_posterior(X)
        log_evidence = logsumexp(log_posterior_unnormalized, axis=1, keepdims=True)
        return log_posterior_unnormalized - log_evidence

    def predict_proba(self, X):
        X = validate_data(self, X, reset=False)
        return np.exp(self.predict_log_proba(X))

    def predict(self, X):
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)
        return self.classes_[np.argmax(self._log_posterior(X), axis=1)]


if __name__ == '__main__':
    from sklearn.datasets import load_digits

    print("Carregando o dataset Digits...")
    digits = load_digits()
    X, y = digits.data, digits.target

    # 2. Dividir os dados em treino e teste
    print("Dividindo os dados em treino e teste (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3. Definir o pipeline
    print("Construindo o pipeline...")
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('gaussian_bayes', GaussianBayesClassifier())
    ])

    # 4. Treinar o pipeline
    print("Treinando o pipeline com os dados de treino...")
    pipeline.fit(X_train, y_train)

    # 5. Fazer predições com o pipeline treinado
    print("Fazendo predições nos dados de teste...")
    y_pred = pipeline.predict(X_test)

    # 6. Avaliar os resultados
    print("\n--- Resultados da Avaliação no Dataset Digits ---")
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Acurácia: {accuracy:.4f}")

    print("\nRelatório de Classificação Detalhado:")
    # Converte os nomes dos alvos para string para o relatório
    target_names = [str(name) for name in digits.target_names]
    report = classification_report(y_test, y_pred, target_names=target_names)
    print(report)