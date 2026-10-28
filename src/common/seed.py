"""Deterministic seeding (same convention as my earlier projects)."""
import os
import random

import numpy as np
import torch


def set_seed(seed: int = 7) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(False)  # CPU conv nondeterminism is fine
    torch.set_num_threads(max(1, os.cpu_count() or 1))
