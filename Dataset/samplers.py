import torch
import random
import numpy as np
from torch.utils.data import Sampler
from typing import Iterator, List, Dict, Optional

class BalancedClassSampler(Sampler):
    """
    Samples elements uniformly from all classes to ensure balanced representation.
    
    This sampler first randomly selects a category_id, then randomly samples an image
    from that category, ensuring even rare classes appear frequently in training.
    
    Args:
        dataset: Dataset with a get_category_idxs method
        num_samples: Number of samples per epoch (defaults to dataset length)
        replacement: Whether to sample with replacement
    """
    
    def __init__(
        self, 
        dataset, 
        num_samples: Optional[int] = None,
        replacement: bool = True
    ):
        self.dataset = dataset
        self.num_samples = num_samples if num_samples is not None else len(dataset)
        self.replacement = replacement
        
        # Build dictionary mapping each category_id to its sample indices
        self.class_indices: Dict[int, List[int]] = {}
        for category_id in dataset.df['category_id'].unique():
            self.class_indices[category_id] = dataset.get_category_idxs(category_id)
            
        self.class_list = list(self.class_indices.keys())
        
    def __iter__(self) -> Iterator[int]:
        # Sample with replacement: continue drawing samples until we have enough
        count = 0
        while count < self.num_samples:
            # Sample a class uniformly at random
            class_id = random.choice(self.class_list)
            
            # Sample an instance from this class
            idx = random.choice(self.class_indices[class_id])
            
            yield idx
            count += 1

    def __len__(self) -> int:
        return self.num_samples
