import numpy as np

class My_MLP:
    def __init__(self, n_hidden=128, activation="relu", lr=0.01, epochs=100, batch_size=32,verbose = False):
        self.n_hidden = n_hidden
        self.activation_name = activation
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose

        self.activations = {
            "sigmoid": (self._sigmoid, self._sigmoid_deriv),
            "relu": (self._relu, self._relu_deriv),
            "cosine": (self._cos, self._cos_deriv),
        }

        if activation not in self.activations:
            raise ValueError("activation must be one of: sigmoid, relu, cosine")

        self.activation, self.activation_derivative = self.activations[activation]

    def _sigmoid(self, Z):
        Z = np.clip(Z, -500, 500)
        return 1 / (1 + np.exp(-Z))

    def _sigmoid_deriv(self, Z):
        s = self._sigmoid(Z)
        return s * (1 - s)

    def _relu(self, Z):
        return np.maximum(0, Z)

    def _relu_deriv(self, Z):
        return (Z > 0).astype(float)

    def _cos(self, Z):
        return np.cos(Z)

    def _cos_deriv(self, Z):
        return -np.sin(Z)

    def fit(self, X_train, y_train, validation_data=None):
        X_train = X_train.values if hasattr(X_train, "values") else X_train
        y_train = y_train.values if hasattr(y_train, "values") else y_train
        y_train = y_train.reshape(-1, 1).astype(float)

        n_samples, n_features = X_train.shape

        self.w1 = np.random.randn(n_features, self.n_hidden) * 0.1
        self.b1 = np.zeros((1, self.n_hidden))

        self.w2 = np.random.randn(self.n_hidden, 1) * 0.1
        self.b2 = np.zeros((1, 1))

        if validation_data is not None:
            X_val, y_val = validation_data
            X_val = X_val.values if hasattr(X_val, "values") else X_val
            y_val = y_val.values if hasattr(y_val, "values") else y_val
            y_val = y_val.reshape(-1, 1).astype(float)

        eps = 1e-5

        for epoch in range(self.epochs):
            perm = np.random.permutation(n_samples)
            train_loss = 0

            for start in range(0, n_samples, self.batch_size):
                idx = perm[start:start + self.batch_size]

                Xb = X_train[idx]
                yb = y_train[idx]

                m = Xb.shape[0]

                Z1 = Xb @ self.w1 + self.b1
                A1 = self.activation(Z1)

                Z2 = A1 @ self.w2 + self.b2
                y_pred = self._sigmoid(Z2)

                loss = -np.mean(yb * np.log(y_pred + eps) + (1 - yb) * np.log(1 - y_pred + eps))

                train_loss += loss * m

                dZ2 = y_pred - yb

                dW2 = (A1.T @ dZ2) / m
                db2 = np.sum(dZ2, axis=0, keepdims=True) / m

                dA1 = dZ2 @ self.w2.T
                dZ1 = dA1 * self.activation_derivative(Z1)

                dW1 = (Xb.T @ dZ1) / m
                db1 = np.sum(dZ1, axis=0, keepdims=True) / m

                self.w1 -= self.lr * dW1
                self.b1 -= self.lr * db1

                self.w2 -= self.lr * dW2
                self.b2 -= self.lr * db2

            train_loss /= n_samples

            if validation_data is not None:
                Z1 = X_val @ self.w1 + self.b1
                A1 = self.activation(Z1)

                Z2 = A1 @ self.w2 + self.b2
                y_val_pred = self._sigmoid(Z2)

                val_loss = -np.mean(y_val * np.log(y_val_pred + eps) + (1 - y_val) * np.log(1 - y_val_pred + eps))
                if self.verbose and epoch%10 == 0: print(f"epoch {epoch} | train_loss {train_loss:.6f} | val_loss {val_loss:.6f}")
            else:
                if self.verbose and epoch%10 == 0: print(f"epoch {epoch} | train_loss {train_loss:.6f}")

    def predict_proba(self, X):
        X = X.values if hasattr(X, "values") else X

        Z1 = X @ self.w1 + self.b1
        A1 = self.activation(Z1)

        Z2 = A1 @ self.w2 + self.b2
        p1 = self._sigmoid(Z2)

        p0 = 1 - p1

        return np.hstack((p0, p1))

    def predict(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)