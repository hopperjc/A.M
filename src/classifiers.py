import numpy as np
from scipy.stats import multivariate_normal, norm
from sklearn.neighbors import KNeighborsClassifier

class BayesianGaussianClassifier:
    def __init__(self):
        self.classes_ = None
        self.means_ = {}
        self.covariances_ = {}
        self.priors_ = {}

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        for c in self.classes_:
            X_c = X[y == c]
            self.means_[c] = np.mean(X_c, axis=0)
            self.covariances_[c] = np.cov(X_c, rowvar=False) + 1e-6 * np.eye(X.shape[1])
            self.priors_[c] = X_c.shape[0] / X.shape[0]

    def predict(self, X):
        predictions = []
        for x in X:
            posteriors = []
            for c in self.classes_:
                likelihood = multivariate_normal.pdf(x, mean=self.means_[c], cov=self.covariances_[c])
                prior = self.priors_[c]
                posteriors.append(prior * likelihood)
            predictions.append(self.classes_[np.argmax(posteriors)])
        return np.array(predictions)

    def predict_proba(self, X):
        proba = []
        for x in X:
            posteriors = []
            for c in self.classes_:
                likelihood = multivariate_normal.pdf(x, mean=self.means_[c], cov=self.covariances_[c])
                prior = self.priors_[c]
                posteriors.append(prior * likelihood)
            posteriors = np.array(posteriors)
            posteriors /= posteriors.sum()  # normalizar
            proba.append(posteriors)
        return np.array(proba)


class BayesianParzenClassifier:
    def __init__(self, h=1.0):
        self.h = h
        self.classes_ = None
        self.data_ = {}
        self.priors_ = {}

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        for c in self.classes_:
            self.data_[c] = X[y == c]
            self.priors_[c] = self.data_[c].shape[0] / X.shape[0]

    def _kernel(self, x, xi):
        return norm.pdf((x - xi) / self.h).prod()

    def predict(self, X):
        predictions = []
        for x in X:
            posteriors = []
            for c in self.classes_:
                density = np.mean([self._kernel(x, xi) for xi in self.data_[c]])
                posteriors.append(density * self.priors_[c])
            predictions.append(self.classes_[np.argmax(posteriors)])
        return np.array(predictions)


class BayesianKNNClassifier:
    def __init__(self, n_neighbors=3, metric="euclidean"):
        self.model = KNeighborsClassifier(n_neighbors=n_neighbors, metric=metric)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)


class MajorityVoteClassifier:
    def __init__(self, classifiers):
        self.classifiers = classifiers

    def fit(self, X, y):
        for clf in self.classifiers:
            clf.fit(X, y)

    def predict(self, X):
        predictions = np.array([clf.predict(X) for clf in self.classifiers])
        final_preds = []
        for i in range(X.shape[0]):
            vals, counts = np.unique(predictions[:, i], return_counts=True)
            final_preds.append(vals[np.argmax(counts)])
        return np.array(final_preds)

    def predict_proba(self, X):
        # Média das probabilidades previstas
        probas = np.mean([clf.predict_proba(X) for clf in self.classifiers if hasattr(clf, 'predict_proba')], axis=0)
        return probas
