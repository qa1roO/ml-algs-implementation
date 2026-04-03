import numpy as np
class Node_Classification:
    def __init__(self, X=None, y=None):
        self.X = np.asarray(X) if X is not None else None
        self.y = np.asarray(y) if y is not None else None
        self.is_leaf = False
        self.prediction = None
        self.feature_index = None
        self.threshold = None
        self.left = None
        self.right = None
        
    def set_leaf(self, y):
        self.is_leaf = True
        counts = np.bincount(y, minlength=2) 
        self.prediction = np.argmax(counts)
        self.probs = counts / counts.sum()


    def split(self, feature_index, threshold):
        assert self.X is not None and self.y is not None
        self.feature_index = feature_index
        self.threshold = threshold
        self.is_leaf = False

        left_indices = self.X[:, feature_index] <= threshold
        right_indices = self.X[:, feature_index] > threshold

        self.left = Node_Classification(self.X[left_indices], self.y[left_indices])
        self.right = Node_Classification(self.X[right_indices], self.y[right_indices])


class MyDesisionTreeClassifier:
    def __init__(self, X = None, y = None, max_depth = 7) -> None:
        self.max_depth = max_depth

    def compute_gini(self, y):
        n = len(y)
        if n == 0:
            return 0
        p0 = sum(y == 0) / n
        p1 = sum(y == 1) / n
        gini = 1 - p0**2 - p1**2
        return gini
    def _find_best_split(self, X, y):
        whole_size = X.shape[0]
        best_gini = float('inf')
        best_feature = None
        best_threshold = None

        total_pos = np.sum(y == 1)

        for feature_index in range(X.shape[1]):
            feat = X[:, feature_index]
            order = np.argsort(feat)
            feat_s = feat[order]
            y_s = y[order]
            diff_idx = np.where(feat_s[:-1] != feat_s[1:])[0]
            if diff_idx.size == 0:
                continue  

            thresholds = (feat_s[diff_idx] + feat_s[diff_idx + 1]) / 2.0

        
            allsum = np.cumsum(y_s == 1)
            left_pos = allsum[diff_idx]              
            left_size = diff_idx + 1                

            right_pos = total_pos - left_pos
            right_size = whole_size - left_size

        
            left_p = left_pos / left_size
            right_p = right_pos / right_size

            left_gini = 1.0 - left_p**2 - (1 - left_p)**2
            right_gini = 1.0 - right_p**2 - (1 - right_p)**2

            weighted = (left_size / whole_size) * left_gini + (right_size / whole_size) * right_gini

            min_idx = np.argmin(weighted)
            min_val = weighted[min_idx]
            if min_val < best_gini:
                best_gini = float(min_val)
                best_feature = feature_index
                best_threshold = float(thresholds[min_idx])

        return best_gini, best_feature, best_threshold


    def _build_tree(self, X, y, depth):
        noda = Node_Classification(X, y)
        if len(np.unique(y)) == 1 or depth == self.max_depth:
            noda.set_leaf(y)
            return noda

        best_gini, best_feature, best_threshold = self._find_best_split(X, y)

        if best_feature is None:
            noda.set_leaf(y)
            return noda

        noda.split(best_feature, best_threshold)
        assert noda.left is not None and noda.right is not None

        noda.left = self._build_tree(noda.left.X, noda.left.y, depth + 1)
        noda.right = self._build_tree(noda.right.X, noda.right.y, depth + 1)

        return noda


    def fit(self, X, y):
        if hasattr(X, "values"):
            X = X.values
        if hasattr(y, "values"):
            y = y.values
        self.root = self._build_tree(X, y, depth=0)
    def _predict_proba_one(self, x, node):
        while not node.is_leaf:
            if x[node.feature_index] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.probs

    def predict_proba(self, X):
        if hasattr(X, "values"):
            X = X.values
        return np.array([self._predict_proba_one(x, self.root) for x in X])

    def predict(self, X):
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)