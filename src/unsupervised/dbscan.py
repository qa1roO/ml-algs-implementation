import numpy as np
from sklearn.neighbors import KDTree

class MyDBScan():
    def __init__(self, eps=0.5, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples

    def fit(self, X):
        n = X.shape[0]

        tree = KDTree(X)
        neighbors = tree.query_radius(X, r=self.eps)

        labels = np.full(n, -1)      
        visited = np.zeros(n, dtype=bool)

        cluster_id = 0

        for i in range(n):

            if visited[i]:
                continue

            visited[i] = True

            if len(neighbors[i]) < self.min_samples:
                continue   

            labels[i] = cluster_id
            queue = list(neighbors[i])

            while queue:
                j = queue.pop()

                if not visited[j]:
                    visited[j] = True

                    if len(neighbors[j]) >= self.min_samples:
                        queue.extend(neighbors[j])

                if labels[j] == -1:
                    labels[j] = cluster_id

            cluster_id += 1

        self.labels_ = labels
        return self