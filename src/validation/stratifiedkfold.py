import numpy as np
class MyStratifiedKFold():
    def __init__(self, n_splits=5, shuffle=False, random_state=None):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def get_n_splits(self):
        return self.n_splits

    def split(self, X, y):
        y = np.array(y)
        unique_classes, _ = np.unique(y, return_inverse=True)
        n_samples = len(y)
        indices = np.arange(n_samples)

        class_indices = {cls: indices[y == cls] for cls in unique_classes}

        if self.shuffle:
            rng = np.random.RandomState(self.random_state)
            for cls in unique_classes:
                rng.shuffle(class_indices[cls])
        tests_per_fold = [[] for _ in range(self.n_splits)]
        for cls in unique_classes:
            splits = np.array_split(class_indices[cls], self.n_splits)
            for fold_idx, part in enumerate(splits):
                tests_per_fold[fold_idx].extend(part.tolist())

        for fold_idx in range(self.n_splits):
            test_idx = np.array(tests_per_fold[fold_idx])
            train_idx = np.setdiff1d(indices, test_idx, assume_unique=True)
            yield train_idx, test_idx