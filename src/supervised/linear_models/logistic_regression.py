import numpy as np

class MyLogisticRegressionSGD:
        def __init__(self, epochs=800, learning_rate=0.01, tol=1e-5):
            self.epochs = epochs
            self.learning_rate = learning_rate
            self.tol = tol
            self.weights = None

        def fit(self, X, y):
            X = X.values if hasattr(X, "values") else X
            y = y.values if hasattr(y, "values") else y

            X_with_bias = np.hstack([X, np.ones((X.shape[0], 1))])
            n_samples, n_features = X_with_bias.shape
            self.weights = np.zeros(n_features)

            for epoch in range(self.epochs):
                prev_weights = self.weights.copy()  
                indices = np.random.permutation(n_samples)
                X_shuffled = X_with_bias[indices]
                y_shuffled = y[indices]

                for i in range(n_samples):
                    x_i = X_shuffled[i]
                    y_i = y_shuffled[i]

                    z = np.dot(x_i, self.weights)
                    sigmoid = 1 / (1 + np.exp(-z))

                    grad = x_i * (sigmoid - y_i)
                    self.weights -= self.learning_rate * grad

                weight_change = np.linalg.norm(self.weights - prev_weights)
                if weight_change < self.tol:
                    print(f"Converged at epoch {epoch+1}")
                    break

        def predict_proba(self, X):
            X = X.values if hasattr(X, "values") else X
            X_with_bias = np.hstack([X, np.ones((X.shape[0], 1))])
            assert self.weights is not None, "Модель еще не обучена, сначала fit, потом predict"
            z = np.dot(X_with_bias, self.weights)
            return 1 / (1 + np.exp(-z))

        def predict(self, X):
            proba = self.predict_proba(X)
            return (proba >= 0.5).astype(int)