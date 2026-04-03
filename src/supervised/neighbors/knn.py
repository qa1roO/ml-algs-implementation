import numpy as np

class MyKNN:
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors

    def fit(self, X, y):
        self.X_train = X.values if hasattr(X, "values") else X
        self.y_train = y.values if hasattr(y, "values") else y

    def _make_predictions(self, x_test_i):
        distances = np.sqrt(np.sum((self.X_train - x_test_i) ** 2, axis=1))
        k_nearest_indexes = np.argpartition(distances, self.n_neighbors)[:self.n_neighbors]
        targets = self.y_train[k_nearest_indexes]
        return np.bincount(targets).argmax()

    def predict(self, X):
        X = X.values if hasattr(X, "values") else X
        return np.array([self._make_predictions(x) for x in X])
    def predict_proba(self, X):
        X = X.values if hasattr(X, "values") else X
        probs = []
        for x in X:
            distances = np.sqrt(np.sum((self.X_train - x)**2, axis=1))
            k_idx = np.argpartition(distances, self.n_neighbors)[:self.n_neighbors]
            targets = self.y_train[k_idx]
            p1 = targets.mean()
            probs.append([1 - p1, p1])
        return np.array(probs)