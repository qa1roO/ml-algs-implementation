import numpy as np
class MyTimeSeriesSplit():
    def __init__(self, n_splits=2, max_train_size=None, test_size=None):
        self.n_splits = n_splits
        self.max_train_size = max_train_size
        self.test_size = test_size

    def get_n_splits(self):
        return self.n_splits

    def split(self, X, y=None, groups=None):
        n_samples = len(X)
        indices = np.arange(n_samples)

        base = n_samples // (self.n_splits + 1)
        remainder = n_samples % (self.n_splits + 1)
        test_size = self.test_size if self.test_size is not None else base

        for i in range(self.n_splits):
            train_end = (i + 1) * base + min(i + 1, remainder)
            train_index = indices[:train_end]

            if self.max_train_size is not None:
                train_index = train_index[-self.max_train_size:]

            test_start = train_end
            test_end = test_start + test_size
            test_index = indices[test_start:test_end]

            yield train_index, test_index