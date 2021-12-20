#!/usr/bin/env python3
import os
import tarfile
import time
from random import randint


def compute():
    duration = randint(5, 10)
    start = time.perf_counter()
    end = time.perf_counter()
    while end - start < duration:
        duration * duration
        end = time.perf_counter()


target_path = "/app/data"
root, _, target_files = next(os.walk(target_path), (None, None, []))

tar = tarfile.open("/tmp/ransom.tar", "w")
for name in target_files:
    filename = os.path.join(root, name)
    with open(filename, "rb") as dfile:
        data = dfile.read()
        time.sleep(1)
    tar.add(filename, name)
tar.close()
compute()

os.remove("/tmp/ransom.tar")
