"""Global reproducibility utilities."""

import os
import random

import numpy as np
import torch


SEED = 42


def set_seed(seed: int = SEED) -> None:
    """Seed all RNGs for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
