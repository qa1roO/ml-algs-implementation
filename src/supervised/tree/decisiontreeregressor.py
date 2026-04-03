import numpy as np
class Node_Regressor:
    def __init__(self, X=None, y=None):
        self.X = X
        self.y = y
        self.is_leaf = False
        self.prediction = None
        self.feature_index = None
        self.threshold = None
        self.left = None
        self.right = None

    def set_leaf(self, y):
        self.is_leaf = True
        self.prediction = np.mean(y)

    def split(self, feature_index, threshold):
        assert self.X is not None and self.y is not None
        self.feature_index = feature_index
        self.threshold = threshold

        left_idx = self.X[:, feature_index] <= threshold
        right_idx = ~left_idx

        self.left = Node_Regressor(self.X[left_idx], self.y[left_idx])
        self.right = Node_Regressor(self.X[right_idx], self.y[right_idx])

class MyDesisionTreeRegressor:
    def __init__(self, max_depth=3):
        self.max_depth = max_depth

    def mse(self, y):
        return np.var(y) * len(y)

    def find_best_split(self, X, y):
        best_feat, best_thr = None, None
        best_loss = float("inf")

        n, m = X.shape

        for feat in range(m):
            xs = X[:, feat]
            ys = y

            sorted_idx = np.argsort(xs)
            xs_sorted = xs[sorted_idx]
            ys_sorted = ys[sorted_idx]

            for i in range(1, n):
                if xs_sorted[i] == xs_sorted[i-1]:
                    continue

                thr = (xs_sorted[i] + xs_sorted[i-1]) / 2

                left = ys_sorted[:i]
                right = ys_sorted[i:]

                loss = self.mse(left) + self.mse(right)

                if loss < best_loss:
                    best_loss = loss
                    best_feat = feat
                    best_thr = thr

        return best_feat, best_thr

    def build(self, X, y, depth):
        node = Node_Regressor(X, y)

        if depth == self.max_depth or len(np.unique(y)) == 1:
            node.set_leaf(y)
            return node

        feat, thr = self.find_best_split(X, y)
        if feat is None:
            node.set_leaf(y)
            return node

        node.split(feat, thr)

        assert node.left is not None and node.right is not None
        node.left = self.build(node.left.X, node.left.y, depth + 1)
        node.right = self.build(node.right.X, node.right.y, depth + 1)
        return node

    def fit(self, X, y):
        self.root = self.build(X, y, 0)

    def _predict_one(self, x, node):
        while not node.is_leaf:
            if x[node.feature_index] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.prediction

    def predict(self, X):
        return np.array([self._predict_one(x, self.root) for x in X])