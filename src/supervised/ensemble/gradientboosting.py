import numpy as np
from ..tree.decisiontreeregressor import MyDesisionTreeRegressor

class MyGBDTClassifier:
    def __init__(self, n_estimators=30, max_depth=3, learning_rate=0.1, max_features=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.max_features = max_features
        self.trees = []
        self.features_indices = []

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _sample_features(self, n_features):
        if self.max_features is None:
            return np.arange(n_features)
        if self.max_features == "sqrt":
            k = int(np.sqrt(n_features))
        elif self.max_features == "log2":
            k = int(np.log2(n_features))
        else:
            k = int(self.max_features)
        return np.random.choice(n_features, k, replace=False)

    def fit(self, X, y):
        if hasattr(X, "values"):
            X = X.values
        if hasattr(y, "values"):
            y = y.values

        n_samples, n_features = X.shape

        # F0 
        p = np.clip(y.mean(), 1e-6, 1 - 1e-6)
        self.F0 = np.log(p / (1 - p))

        F = np.full(n_samples, self.F0)

        for m in range(self.n_estimators):

            prob = self._sigmoid(F)

            r = y - prob   

            feats = self._sample_features(n_features)
            self.features_indices.append(feats)

            tree = MyDesisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X[:, feats], r)
            self.trees.append(tree)

            F += self.learning_rate * tree.predict(X[:, feats])

    def predict_proba(self, X):
        if hasattr(X, "values"):
            X = X.values

        F = np.full(len(X), self.F0)

        for feats, tree in zip(self.features_indices, self.trees):
            F += self.learning_rate * tree.predict(X[:, feats])

        proba = self._sigmoid(F)
        return np.vstack([1 - proba, proba]).T

    def predict(self, X):
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)