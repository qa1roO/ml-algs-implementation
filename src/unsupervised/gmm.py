import numpy as np

class MyGMM:
    def __init__(self, n_components, max_iter=100, tol=1e-4):
        self.K = n_components
        self.max_iter = max_iter
        self.tol = tol

    def _gaussian(self, X, mean, cov):
        d = X.shape[1]
        cov_inv = np.linalg.inv(cov)
        det = np.linalg.det(cov)

        diff = X - mean
        exponent = -0.5 * np.sum(diff @ cov_inv * diff, axis=1)

        coef = 1 / np.sqrt((2 * np.pi) ** d * det)
        return coef * np.exp(exponent)

    def fit(self, X):
        n, d = X.shape

        self.means = X[np.random.choice(n, self.K, replace=False)]
        self.covs = np.array([np.eye(d) for _ in range(self.K)])
        self.weights = np.ones(self.K) / self.K

        log_likelihood_old = 0

        for _ in range(self.max_iter):

            probs = np.zeros((n, self.K))

            for k in range(self.K):
                probs[:, k] = self.weights[k] * self._gaussian(
                    X, self.means[k], self.covs[k]
                )

            responsibilities = probs / probs.sum(axis=1, keepdims=True)

            Nk = responsibilities.sum(axis=0)

            self.weights = Nk / n

            self.means = (responsibilities.T @ X) / Nk[:, None]

            for k in range(self.K):
                diff = X - self.means[k]
                self.covs[k] = (
                    responsibilities[:, k][:, None] * diff
                ).T @ diff / Nk[k]

                self.covs[k] += 1e-6 * np.eye(d)

            log_likelihood = np.sum(np.log(probs.sum(axis=1)))

            if np.abs(log_likelihood - log_likelihood_old) < self.tol:
                break

            log_likelihood_old = log_likelihood

        self.responsibilities_ = responsibilities
        self.labels_ = np.argmax(responsibilities, axis=1)

        return self