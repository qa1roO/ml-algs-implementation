import numpy as np
from ..tree.decisiontreeclassifier import MyDesisionTreeClassifier

class MyRandomForestClassifier:
    def __init__(self, n_estimators=50, max_depth=7, max_features='sqrt', random_state=21):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.random_state = random_state
        self.trees = []
        self.features_indices = []
        np.random.seed(random_state)

    def _sample_features(self, n_features):
        if self.max_features == "sqrt":
            k = int(np.sqrt(n_features))
        elif self.max_features == "log2":
            k = int(np.log2(n_features))
        else:
            k = n_features
        return np.random.choice(n_features, k, replace=False)

    def _bootstrap_sample(self, X, y):
        n = len(X)
        indices = np.random.choice(n, n, replace=True)
        return X[indices], y[indices]

    def fit(self, X, y):
        if hasattr(X, "values"):
            X = X.values
        if hasattr(y, "values"):
            y = y.values

        n_features = X.shape[1]

        for i in range(self.n_estimators):
            Xb, yb = self._bootstrap_sample(X, y)

            feats = self._sample_features(n_features)
            self.features_indices.append(feats)

            tree = MyDesisionTreeClassifier(max_depth=self.max_depth)
            tree.fit(Xb[:, feats], yb)
            self.trees.append(tree)

    def predict_proba(self, X):
        if hasattr(X, "values"):
            X = X.values

        probs = np.zeros((len(X), 2))
        for feats, tree in zip(self.features_indices, self.trees):
            probs += tree.predict_proba(X[:, feats])

        probs /= len(self.trees)
        return probs

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)