import numpy as np
class MyKFold():
    def __init__(self, n_splits=5, shuffle = False, random_state = None):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
    def get_n_splits(self):
        return self.n_splits
    def split(self, X):
        n = len(X)
        indices = np.arange(n)
        if self.shuffle:
            rng = np.random.RandomState(self.random_state)
            rng.shuffle(indices)
        fold_sizes = [n//self.n_splits]*self.n_splits
        for i in range(n%self.n_splits):
            fold_sizes[i]+=1
        current = 0
        for idx, fold_size in enumerate(fold_sizes):
            start, stop = current, current+fold_size
            test_index = indices[start:stop]
            train_index = np.concatenate([indices[:start],indices[stop:]])
            current = stop
            yield train_index, test_index