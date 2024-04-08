import random
import subprocess
from typing import Callable
import time

import gymnasium
import numpy as np
import torch


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def timeit(func: Callable, *args, **kwargs):
    start = time.time()
    result = func(*args, **kwargs)
    print(f'Function {func.__name__} takes {time.time() - start:.2f} seconds')
    return result


# kill all wandb processes – sometimes required due a bug in wandb
def kill_all_wandb_processes():
    subprocess.run('ps aux|grep wandb|grep -v grep | awk \'{print $2}\'|xargs kill -9', shell=True)
