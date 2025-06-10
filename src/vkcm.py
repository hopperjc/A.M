import numpy as np
from sklearn.metrics import pairwise_distances, silhouette_score, adjusted_rand_score
from sklearn.utils import check_random_state


class VKCM:
    def __init__(self, n_clusters=2, max_iter=100, tol=1e-4, gamma=1.0, random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.gamma = gamma  # parâmetro do kernel Gaussiano
        self.random_state = check_random_state(random_state)

    def _gaussian_kernel(self, x, y, w):
        diff = (x - y) ** 2
        weighted_diff = np.dot(w, diff)
        return np.exp(-self.gamma * weighted_diff)

    def _compute_kernel_matrix(self, X, w):
        n_samples = X.shape[0]
        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(i, n_samples):
                k_val = self._gaussian_kernel(X[i], X[j], w)
                K[i, j] = K[j, i] = k_val
        return K

    def fit(self, X):
        n_samples, n_features = X.shape
        w = np.ones(n_features)
        labels = self.random_state.randint(self.n_clusters, size=n_samples)
        prev_labels = np.copy(labels)

        for iteration in range(self.max_iter):
            centroids = np.array([X[labels == i].mean(axis=0) if np.any(labels == i) else self.random_state.rand(n_features)
                                   for i in range(self.n_clusters)])

            variances = np.array([X[labels == i].var(axis=0) + 1e-6 if np.any(labels == i) else np.ones(n_features)
                                   for i in range(self.n_clusters)])
            w = 1 / (variances.mean(axis=0))
            w = w / w.sum()

            new_labels = np.zeros(n_samples, dtype=int)
            for i in range(n_samples):
                distances = [1 - self._gaussian_kernel(X[i], centroids[j], w) for j in range(self.n_clusters)]
                new_labels[i] = np.argmin(distances)

            if np.linalg.norm(new_labels - prev_labels) < self.tol:
                break
            prev_labels = new_labels

        self.labels_ = new_labels
        self.relevance_weights_ = w
        return self

    def predict(self, X):
        return self.labels_

    def score(self, X):
        return silhouette_score(X, self.labels_)

    def adjusted_rand_index(self, y_true):
        return adjusted_rand_score(y_true, self.labels_)


def evaluate_varying_k(X, y_true=None, k_values=[2, 3, 4, 5], runs=50, gamma=1.0):
    results = {}
    for k in k_values:
        silhouette_scores = []
        label_sets = []
        best_score = -1
        best_model = None

        for _ in range(runs):
            model = VKCM(n_clusters=k, gamma=gamma, random_state=np.random.randint(10000))
            model.fit(X)
            score = model.score(X)
            silhouette_scores.append(score)
            label_sets.append(model.labels_)
            if score > best_score:
                best_score = score
                best_model = model

        results[k] = {
            "silhouette_scores": silhouette_scores,
            "avg_silhouette": np.mean(silhouette_scores),
            "best_labels": best_model.labels_,
            "relevance_weights": best_model.relevance_weights_,
            "ari": best_model.adjusted_rand_index(y_true) if y_true is not None else None
        }

    best_k = max(results, key=lambda k: results[k]["avg_silhouette"])
    return best_k, results[best_k], results



