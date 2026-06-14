import torch
import numpy as np
from torch.utils.data import Sampler
from typing import List, Dict

class BalancedBatchSampler(Sampler):
    """
    Custom PyTorch Sampler to generate class-balanced batches.
    Ensures each batch contains an equal number of samples per class.
    """
    def __init__(self, labels: np.ndarray, num_classes: int, num_samples_per_class: int):
        super().__init__(labels)
        self.labels = labels
        self.num_classes = num_classes
        self.num_samples_per_class = num_samples_per_class
        self.batch_size = self.num_classes * self.num_samples_per_class

        self.class_indices: Dict[int, List[int]] = {c: [] for c in range(self.num_classes)}
        for i, label in enumerate(self.labels):
            self.class_indices[label].append(i)

        self.class_iterators: Dict[int, iter] = {}
        for c in range(self.num_classes):
            np.random.shuffle(self.class_indices[c])
            self.class_iterators[c] = iter(self.class_indices[c])

        min_class_size = min(len(v) for v in self.class_indices.values())
        self.num_batches = min_class_size // self.num_samples_per_class

    def _get_next_index(self, class_id: int) -> int:
        try:
            return next(self.class_iterators[class_id])
        except StopIteration:
            np.random.shuffle(self.class_indices[class_id])
            self.class_iterators[class_id] = iter(self.class_indices[class_id])
            return next(self.class_iterators[class_id])

    def __iter__(self):
        for _ in range(self.num_batches):
            batch_indices = []
            for c in range(self.num_classes):
                for _ in range(self.num_samples_per_class):
                    idx = self._get_next_index(c)
                    batch_indices.append(idx)
            yield batch_indices

    def __len__(self) -> int:
        return self.num_batches