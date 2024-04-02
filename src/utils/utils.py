import random
from typing import Callable
import time

import numpy as np
import torch


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def timeit(func: Callable, *args, **kwargs):
    start = time.time()
    result = func(*args, **kwargs)
    print(f'Function {func.__name__} takes {time.time() - start:.2f} seconds')
    return result
