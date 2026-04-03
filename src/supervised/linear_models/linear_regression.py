import numpy as np
import random

class MyLinearRegression():
    def __init__(self, epochs=1000):
        self.epochs = epochs
        self.bias = None
        self.weights = None

    def fit(self, X_train, y_train, learning_rate=0.01, solution = 'SGD', batch_size =32):
        if solution=='SGD':
            self.bias = 0
            self.weights = np.zeros(X_train.shape[1])
            for _ in range(self.epochs):
                i = random.randint(0, len(X_train) - 1)
                features = X_train.iloc[i].values
                target = y_train.iloc[i]
                self.weights, self.bias = self.square_trick(target, features, learning_rate)
        elif solution=='analytical':
            X_b = np.hstack([np.ones((X_train.shape[0], 1)), X_train.values]) 
            theta = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y_train
            self.bias = theta[0]
            self.weights = theta[1:]
        elif solution == 'GD':
            self.bias = 0
            self.weights = np.zeros(X_train.shape[1])
            for _ in range(self.epochs):
                predictions = self.bias + np.dot(X_train.values, self.weights)
                errors = y_train.values - predictions
                weight_grad = -2 * np.dot(X_train.values.T, errors) / len(X_train)
                bias_grad = -2 * np.mean(errors)
                self.weights -= learning_rate * weight_grad
                self.bias -= learning_rate * bias_grad
        elif solution == 'mini_batch':
            self.bias = 0
            self.weights = np.zeros(X_train.shape[1])
            n_samples = X_train.shape[0]
            for _ in range(self.epochs):
                indices = np.random.permutation(n_samples)
                X_shuffled = X_train.values[indices]
                y_shuffled = y_train.values[indices]
                for start in range(0, n_samples, batch_size):
                    end = start + batch_size
                    X_batch = X_shuffled[start:end]
                    y_batch = y_shuffled[start:end]

                    predictions = self.bias + np.dot(X_batch, self.weights)
                    errors = y_batch - predictions

                    weight_grad = -2 * np.dot(X_batch.T, errors) / len(X_batch)
                    bias_grad = -2 * np.mean(errors)

                    self.weights -= learning_rate * weight_grad
                    self.bias -= learning_rate * bias_grad


    def predict(self, X):
        assert self.weights is not None and self.bias is not None, "Модель еще не обучена, сначала fit, потом predict"
        return self.bias + np.dot(X, self.weights)

    def square_trick(self, target, features, learning_rate=0.01):
        assert self.weights is not None and self.bias is not None
        prediction = self.bias + np.dot(self.weights, features)
        error = target - prediction
        self.weights += learning_rate * features * error
        self.bias += learning_rate * error
        return self.weights, self.bias