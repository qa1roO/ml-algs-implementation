import numpy as np

class NB():
    def __init__(self):
        pass

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        X_0 = X[y == self.classes_[0]]
        X_1 = X[y == self.classes_[1]]
        self.E_0 = X_0.mean(axis=0)
        self.E_1 = X_1.mean(axis=0)
        self.Var_0 = X_0.var(axis=0, ddof=0)
        self.Var_1 = X_1.var(axis=0, ddof=0)
        self.P_0 = len(X_0) / len(X)
        self.P_1 = len(X_1) / len(X)

    def _calc(self, x, mean, std):
        return (1 / (np.sqrt(2 * np.pi) * std)) * np.exp(-0.5 * ((x - mean) / std) ** 2)

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def predict_proba(self, X):
        X = np.asarray(X)
        std0 = np.sqrt(self.Var_0)
        std1 = np.sqrt(self.Var_1)

        log_p0 = np.log(self.P_0) + np.sum(np.log(self._calc(X, self.E_0, std0)), axis=1)
        log_p1 = np.log(self.P_1) + np.sum(np.log(self._calc(X, self.E_1, std1)), axis=1)

        max_p = np.maximum(log_p0, log_p1)
        p0_exp = np.exp(log_p0 - max_p)
        p1_exp = np.exp(log_p1 - max_p)
        total = p0_exp + p1_exp

        return np.column_stack([p0_exp / total, p1_exp / total])
