import numpy as np
class MyGroupKFold():
    def __init__(self, n_splits):
        self.n_splits = n_splits
    def get_n_splits(self):
        return self.n_splits
    def split(self, X, groups):
        groups = np.array(groups)
        unique_groups = np.unique(groups)
        n_groups = len(unique_groups)

        if self.n_splits > n_groups:
            raise ValueError(f"Слишком мало групп ({n_groups}) для {self.n_splits} фолдов.")

        fold_sizes = [n_groups // self.n_splits] * self.n_splits
        for i in range(n_groups % self.n_splits):
            fold_sizes[i] += 1

        current = 0
        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            test_groups = unique_groups[start:stop]

            test_mask = np.isin(groups, test_groups)
            test_idx = np.where(test_mask)[0]
            train_idx = np.where(~test_mask)[0]

            yield train_idx, test_idx

            current = stop
            