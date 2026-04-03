import numpy as np
from collections import Counter

class ExtraTree:
    def __init__(self, max_depth=None, min_samples_split=2, max_features=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.tree = None

    def fit(self, X, y):
        self.tree = self._build(X, y, depth=0)

    def _build(self, X, y, depth):
        if (self.max_depth is not None and depth >= self.max_depth) \
           or len(y) < self.min_samples_split \
           or len(np.unique(y)) == 1:
            return Counter(y).most_common(1)[0][0]

        n_samples, n_features = X.shape

        m = self.max_features or int(np.sqrt(n_features))
        features = np.random.choice(n_features, m, replace=False)

        feature = np.random.choice(features)

        values = X[:, feature]
        threshold = np.random.uniform(values.min(), values.max())

        left_mask = X[:, feature] <= threshold

        if left_mask.sum() == 0 or left_mask.sum() == len(y):
            return Counter(y).most_common(1)[0][0]

        return {
            "feature": feature,
            "threshold": threshold,
            "left": self._build(X[left_mask], y[left_mask], depth + 1),
            "right": self._build(X[~left_mask], y[~left_mask], depth + 1),
        }

    def _predict_one(self, x, node):
        if not isinstance(node, dict):
            return node

        if x[node["feature"]] <= node["threshold"]:
            return self._predict_one(x, node["left"])
        else:
            return self._predict_one(x, node["right"])

    def predict(self, X):
        return np.array([self._predict_one(row, self.tree) for row in X])
    
class ExtraTreesClassifier:
    def __init__(self, n_estimators=50, max_depth=None, min_samples_split=2, max_features=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.trees = []

    def fit(self, X, y):
        self.trees = []

        for _ in range(self.n_estimators):
            tree = ExtraTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=self.max_features
            )
            tree.fit(X, y)
            self.trees.append(tree)

    def predict(self, X):
        preds = np.array([tree.predict(X) for tree in self.trees])
        return (preds.mean(axis=0) >= 0.5).astype(int)

    def predict_proba(self, X):
        preds = np.array([tree.predict(X) for tree in self.trees])
        p = preds.mean(axis=0)
        return np.vstack([1-p, p]).T