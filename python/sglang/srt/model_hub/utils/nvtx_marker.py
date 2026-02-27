

import torch.cuda.nvtx as nvtx
import functools

import time
import logging
from contextlib import contextmanager
from typing import Generator

def nvtx_profile(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with nvtx.range(func.__name__):
            return func(*args, **kwargs)
    return wrapper

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

@contextmanager
def count_time(tag: str = "Operation") -> Generator[None, None, None]:
    start_time = time.perf_counter()
    try:
        yield
    finally:
        elapsed_time = time.perf_counter() - start_time
        # logger.info(f"[{tag}] completed in {elapsed_time:.6f}s")
