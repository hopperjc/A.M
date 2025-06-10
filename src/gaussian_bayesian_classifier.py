import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
from scipy.linalg import LinAlgError
import warnings
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels


# --- Classe do Classificador Bayesiano Gaussiano ---
class GaussianBayesClassifier(BaseEstimator, ClassifierMixin):
    """
    Classificador Bayesiano Gaussiano com MLE para parâmetros.

    Assume que a densidade condicional de classe p(x|omega_i) segue
    uma distribuição Normal Multivariada. Estima os parâmetros (média,
    covariância) usando Máxima Verossimilhança (MLE) e os priors P(omega_i)
    pela frequência das classes.

    Parâmetros:
    ----------
    reg_epsilon : float, default=1e-6
        Valor adicionado à diagonal da matriz de covariância para
        regularização e estabilidade numérica.

    Atributos:
    ---------
    classes_ : ndarray de shape (n_classes,)
        Rótulos das classes únicas.
    priors_ : dict
        Probabilidades a priori P(omega_i) para cada classe.
    means_ : dict
        Vetores de média (MLE) para cada classe.
    inv_covariances_ : dict
        Inversas das matrizes de covariância (MLE, regularizadas) para cada classe.
    log_det_covariances_ : dict
        Log do determinante das matrizes de covariância (MLE, regularizadas)
        para cada classe.
    n_features_in_ : int
        Número de features vistas durante o fit.
    """
    def __init__(self, reg_epsilon=1e-6):
        self.reg_epsilon = reg_epsilon

    def fit(self, X, y):
        """
        Treina o classificador estimando os parâmetros a partir dos dados.

        Parâmetros:
        ----------
        X : array-like de shape (n_samples, n_features)
            Vetores de treinamento.
        y : array-like de shape (n_samples,)
            Valores alvo (rótulos das classes).

        Retorna:
        -------
        self : object
            Instância ajustada do classificador.
        """
        # Validação de entrada
        X, y = check_X_y(X, y)

        self.classes_ = unique_labels(y)
        self.n_features_in_ = X.shape[1]
        n_samples = X.shape[0]

        # Dicionários para armazenar parâmetros por classe
        self.priors_ = {}  # Probabilidades a priori P(ω_i)
        self.means_ = {}  # Vetores de média μ_i
        self.inv_covariances_ = {}  # Inversas das matrizes de covariância Σ_i^-1
        self.log_det_covariances_ = {}  # Log do determinante de Σ_i

        for c in self.classes_:
            X_c = X[y == c]
            n_samples_c = X_c.shape[0]

            # Prior (MLE)
            self.priors_[c] = n_samples_c / n_samples

            # Média (MLE)
            self.means_[c] = np.mean(X_c, axis=0)

            # Covariância (MLE) e Regularização
            if n_samples_c <= self.n_features_in_:
                warnings.warn(f"Classe {c} tem {n_samples_c} amostras e {self.n_features_in_} features. "
                              "A matriz de covariância pode ser singular. Aplicando regularização.")

            cov = np.cov(X_c, rowvar=False)
            cov_reg = cov + np.eye(self.n_features_in_) * self.reg_epsilon

            try:
                self.inv_covariances_[c] = np.linalg.inv(cov_reg)
                sign, logdet = np.linalg.slogdet(cov_reg)
                if sign <= 0:
                     raise LinAlgError("Log-determinante não é positivo após regularização.")
                self.log_det_covariances_[c] = logdet
            except LinAlgError as e:
                print(f"Erro Crítico: Matriz de covariância para a classe {c} é singular e não invertível, mesmo com regularização {self.reg_epsilon}. Detalhes: {e}")
                # Tratar o erro - talvez parar, ou usar uma covariância diagonal (Naive Bayes) como fallback?
                # Por simplicidade, vamos relançar o erro por enquanto.
                raise e

        return self

    def _log_likelihood(self, X):
        """Calcula o log da verossimilhança log(p(x|omega_i)) para todas as classes."""
        check_is_fitted(self)
        X = check_array(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"Número de features esperado {self.n_features_in_}, mas recebeu {X.shape[1]}")

        log_likelihoods = np.zeros((X.shape[0], len(self.classes_)))

        for i, c in enumerate(self.classes_):
            mean = self.means_[c]
            inv_cov = self.inv_covariances_[c]
            log_det_cov = self.log_det_covariances_[c]

            # Constante da PDF log normal multivariada
            log_pdf_constant = -0.5 * self.n_features_in_ * np.log(2 * np.pi) - 0.5 * log_det_cov

            # Calcular termo do expoente para todas as amostras de X de uma vez
            diff = X - mean  # Shape (n_samples, n_features)
            # Mahalanobis^2: (X-mu) @ Sigma^-1 @ (X-mu).T -> precisamos diagonal
            # (X-mu) @ Sigma^-1 -> shape (n_samples, n_features)
            # ((X-mu) @ Sigma^-1) * (X-mu) -> element-wise product, shape (n_samples, n_features)
            # Sum along feature axis (axis=1)
            mahalanobis_sq = np.sum(np.dot(diff, inv_cov) * diff, axis=1)
            exponent_term = -0.5 * mahalanobis_sq

            log_likelihoods[:, i] = log_pdf_constant + exponent_term

        return log_likelihoods

    def _log_posterior(self, X):
        """Calcula o log da probabilidade a posteriori (não normalizado)."""
        log_likelihood = self._log_likelihood(X)  # Shape (n_samples, n_classes)
        log_priors = np.log(np.array([self.priors_[c] for c in self.classes_]))  # Shape (n_classes,)

        # log P(omega|x) ~ log p(x|omega) + log P(omega)
        log_posterior_unnormalized = log_likelihood + log_priors  # Broadcasting aplica log_priors a cada linha

        return log_posterior_unnormalized

    def predict_log_proba(self, X):
        """
        Calcula o logaritmo das probabilidades a posteriori P(omega_i|x).
        (Normaliza para que a soma das probabilidades seja 1).
        """
        log_posterior_unnormalized = self._log_posterior(X)  # Shape (n_samples, n_classes)

        # Normalização: P(omega_i|x) = exp(log_posterior_i) / sum_j(exp(log_posterior_j))
        # log P(omega_i|x) = log_posterior_i - logsumexp(log_posterior_all_classes)
        from scipy.special import logsumexp
        log_evidence = logsumexp(log_posterior_unnormalized, axis=1, keepdims=True)  # Shape (n_samples, 1)
        log_proba = log_posterior_unnormalized - log_evidence

        return log_proba

    def predict_proba(self, X):
        """
        Calcula as probabilidades a posteriori P(omega_i|x).
        """
        return np.exp(self.predict_log_proba(X))

    def predict(self, X):
        """
        Prevê a classe para as amostras em X.
        """
        check_is_fitted(self)
        X = check_array(X)

        # Encontra a classe com a maior log-probabilidade a posteriori
        log_posterior = self._log_posterior(X)  # (n_samples, n_classes)
        indices_max_posterior = np.argmax(log_posterior, axis=1)  # (n_samples,)

        # Mapeia os índices de volta para os rótulos das classes
        predictions = self.classes_[indices_max_posterior]

        return predictions

