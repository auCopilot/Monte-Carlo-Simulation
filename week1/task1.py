import numpy as np
import statistics as st
import time
import random

from numpy.f2py.crackfortran import endifs


def track_runtime(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        t = end - start
        return t, result
    return wrapper

ns = [10**i for i in range(8)]

# (i)
print("(i)")
@track_runtime
def one_at_a_time(n):
    i = 0
    for _ in range(n):
        number = random.randint(0,1)
        if number > 0.3:
            i += 1
    return i / n

for n in ns:
    t, result = one_at_a_time(n)
    print(f"Runtime, n = {n}: ", t)

# (ii)
print("(ii)")
@track_runtime
def all_at_once(n):
    numbers = np.random.uniform(0,1,n)
    i = 0
    for v in numbers:
        i += 1 if v > 0.3 else 0
    return i / n

for n in ns:
    t, result = all_at_once(n)
    print(f"Runtime, n = {n}: ", t)

#(iii)
print("(iii)")
@track_runtime
def vectorized(n):
    numbers = np.random.uniform(0,1,n) > 0.3
    return np.sum(numbers) / n

for n in ns:
    t, result = vectorized(n)
    print(f"Runtime, n = {n}: ", t)


