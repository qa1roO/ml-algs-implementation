import numpy as np
class MyRidge():
    def __init__(self, epochs=1000):
        self.epochs = epochs
        self.bias = None
        self.weights = None

    def fit(self, X_train, y_train, lmb=0.01, solution='analytical', learning_rate = 0.001):
        if solution =='analytical':
            X_b = np.hstack([np.ones((X_train.shape[0], 1)), X_train.values])
            reg_matrix = np.eye(X_b.shape[1])
            reg_matrix[0, 0] = 0 
            theta = np.linalg.inv(X_b.T @ X_b + lmb * reg_matrix) @ X_b.T @ y_train
            self.bias = theta[0]
            self.weights = theta[1:]
        elif solution == 'GD':
            self.bias = 0
            self.weights = np.zeros(X_train.shape[1])
            for _ in range(self.epochs):
                predictions = self.bias + np.dot(X_train.values, self.weights)
                errors = y_train.values - predictions
                weight_grad = (-2 * np.dot(X_train.values.T, errors) / len(X_train)) + 2 * lmb * self.weights
                bias_grad = -2 * np.mean(errors)
                self.weights -= learning_rate * weight_grad
                self.bias -= learning_rate * bias_grad

    def predict(self, X):
        assert self.weights is not None and self.bias is not None, "Модель еще не обучена, сначала fit, потом predict"
        return self.bias + np.dot(X, self.weights)