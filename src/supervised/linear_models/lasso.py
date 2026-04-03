import numpy as np
class MyLasso():
    def __init__(self, epochs=1000):
        self.epochs = epochs
        self.bias = None
        self.weights = None

    def fit(self, X_train, y_train, lmb=0.01, learning_rate = 0.001):
        X_train = np.asarray(X_train)
        y_train = np.asarray(y_train)
        self.bias = 0
        self.weights = np.zeros(X_train.shape[1])
        for _ in range(self.epochs):
            n=len(y_train)
            predictions = self.bias + np.dot(X_train, self.weights)
            errors = y_train - predictions
            gradient = -2/n * X_train.T @ errors
            w_tmp = self.weights - learning_rate * gradient
            self.weights = np.sign(w_tmp) * np.maximum(0, np.abs(w_tmp) - learning_rate * lmb)
            bias_grad = -2 * np.mean(errors)    
            self.bias -= learning_rate * bias_grad  

    def predict(self, X):
        assert self.weights is not None and self.bias is not None, "Модель еще не обучена, сначала fit, потом predict"
        return self.bias + np.dot(X, self.weights)