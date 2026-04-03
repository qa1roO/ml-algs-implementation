import numpy as np

class MyKMeans():
    def __init__(self, n_clusters, random_state=None, max_iter=100):
        self.K = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state

    def fit(self, X):
        rng = np.random.default_rng(self.random_state)

        n_samples, n_features = X.shape

        self.centroids = X[rng.choice(n_samples, self.K, replace=False)]

        for _ in range(self.max_iter):
            distances = np.linalg.norm(X[:, np.newaxis, :] - self.centroids, axis=2)
            labels = np.argmin(distances, axis=1)

            new_centroids = np.zeros_like(self.centroids)
            for k in range(self.K):
                members = X[labels == k]
                if len(members) == 0:
                    new_centroids[k] = X[rng.choice(n_samples)]
                else:
                    new_centroids[k] = members.mean(axis=0)

            if np.allclose(self.centroids, new_centroids, atol=1e-8):
                break

            self.centroids = new_centroids

        self.labels_ = labels
        return self
    def predict(self, X):
        distances = np.linalg.norm(X[:, np.newaxis, :] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)
    def distortion(self, X):
        distances = np.linalg.norm(X - self.centroids[self.labels_], axis=1)
        return np.sum(distances ** 2)