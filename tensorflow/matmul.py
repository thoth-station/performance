#!/usr/bin/env python3

# Copyright 2018 Jason Zaman <jason AT perfinion.com> 2018
# Licensed under Apache-2
#
# Additional changes for project Thoth by Thoth team.
#

import argparse
import os
import sys
import numpy as np
import json
import tensorflow as tf
from timeit import time


# Datatype used.
# Options:
#   float16
#   float32
#   float64
#   int32
_ARGS_DTYPE = os.getenv('MATMUL_DTYPE', 'float32')
if _ARGS_DTYPE not in ('float16', 'float32', 'float64', 'int32'):
    raise ValueError("Unknown MATMUL_DTYPE")
print("DTYPE set to %s" % _ARGS_DTYPE, file=sys.stderr)

# Run on CPU or GPU.
# Options:
#   cpu
#   gpu
_ARGS_DEVICE = os.getenv('MATMUL_DEVICE', 'cpu')
print("DEVICE set to %s" % _ARGS_DEVICE, file=sys.stderr)

# Number of repetitions.
# Options:
#   A pozitive integer.
_ARGS_REPS = int(os.getenv('MATMUL_REPS', 5))
print("REPS set to %s" % _ARGS_REPS, file=sys.stderr)


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
if _ARGS_DEVICE == 'cpu':
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def bench(n):
    tf.reset_default_graph()
    with tf.device("/%s:0" % (_ARGS_DEVICE)):
        matrix1 = tf.Variable(tf.ones((n, n), dtype=_ARGS_DTYPE))
        matrix2 = tf.Variable(tf.ones((n, n), dtype=_ARGS_DTYPE))
        product = tf.matmul(matrix1, matrix2)

    times = []
    config = tf.ConfigProto()
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        # warmup
        sess.run(product.op)

        for i in range(_ARGS_REPS):
            start = time.time()
            sess.run(product.op)
            times.append(time.time() - start)

    times_ms = 1000 * np.array(times)  # in seconds, convert to ms
    elapsed_ms = np.median(times_ms)

    ops = n ** 3 + (n - 1) * n ** 2  # n^2*(n-1) additions, n^3 multiplications
    rate = ops / elapsed_ms / 10 ** 6  # in GFLOPS. (/ milli / 10**6) == (/ 10 ** 9)
    print('%d x %d matmul took:   \t%.4f ms,\t %.2f GFLOPS' % (n, n, elapsed_ms, rate,), file=sys.stderr)
    return rate, elapsed_ms


def main():
    np.set_printoptions(suppress=True)
    print("# Version: %s, path: %s" % (tf.__version__, tf.__path__), file=sys.stderr)
    print("size,time,flop", file=sys.stderr)

    result = {}
    for i in range(8, 15):  # [256 ... 16384]
        n = 2 ** i
        rate, elapsed_ms = bench(n)

        result[str(n)] = {
            "elapsed_ms": elapsed_ms,
            "rate": rate
        }

    json.dump(result, sys.stdout, indent=2)

if __name__ == '__main__':
    main()
