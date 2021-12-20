#!/usr/bin/env python3
import random
import numpy as np
from loguru import logger
from multiprocessing import Pool, cpu_count


def generate(i):
    logger.info("{} creating array", i)
    array = np.random.rand(random.randint(200, 250), 200, 200, random.randint(1, 5))
    logger.info("{} saving file {} Mbytes", i, array.nbytes / 1e6)
    np.savez(f"data/{i}.npz", array)


if __name__ == "__main__":
    with Pool(cpu_count()) as p:
        p.map(generate, range(10))
