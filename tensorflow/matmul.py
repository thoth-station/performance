#!/usr/bin/env python3

# Copyright 2018 Jason Zaman <jason AT perfinion.com> 2018
# Licensed under Apache-2
#
# Additional changes for project Thoth by Thoth team.
#

import logging
import os
import sys
import numpy as np
import json
import tensorflow as tf
from timeit import time

_LOGGER = logging.getLogger(__name__)


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
#   A positive integer.
_ARGS_REPS = int(os.getenv('MATMUL_REPS', 2000))
print("REPS set to %s" % _ARGS_REPS, file=sys.stderr)

# Size of matrix.
# Options:
#   A positive integer.
_ARGS_MATRIX_SIZE = int(os.getenv('MATMUL_MATRIX_SIZE', 512))
print("MATRIX size set to %s" % _ARGS_MATRIX_SIZE, file=sys.stderr)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
if _ARGS_DEVICE == 'cpu':
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def _get_aicoe_tensorflow_build_info():
    """Try to obtain information of AICoE TensorFlow builds.

    Do whatever is needed in this function, if there is an error, the reported build information is
    set to None (e.g. AICoE TensorFlow is not installed and such).
    """
    try:
        path = os.path.dirname(os.path.dirname(tf.__file__))
        build_info_path = os.path.join(path, 'tensorflow-' + tf.__version__ + '.dist-info', 'build_info.json')
        with open(build_info_path, 'r') as build_info_file:
            build_info = json.load(build_info_file)
        return build_info
    except Exception:
        _LOGGER.exception("Failed to obtain AICoE specific build information for TensorFlow")

    return None


def bench_v1(n: int):
    times = []
    tf.reset_default_graph()
    with tf.device("/%s:0" % (_ARGS_DEVICE)):
        matrix1 = tf.Variable(tf.ones((n, n), dtype=_ARGS_DTYPE))
        matrix2 = tf.Variable(tf.ones((n, n), dtype=_ARGS_DTYPE))
        product = tf.matmul(matrix1, matrix2)

    config = tf.ConfigProto()
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        # warmup
        sess.run(product.op)

        for i in range(_ARGS_REPS):
            start = time.monotonic()
            sess.run(product.op)
            times.append(time.monotonic() - start)

    times_ms = 1000 * np.array(times)  # in seconds, convert to ms
    elapsed_ms = np.median(times_ms)

    ops = n ** 3 + (n - 1) * n ** 2  # n^2*(n-1) additions, n^3 multiplications
    rate = ops / elapsed_ms / 10 ** 6  # in GFLOPS. (/ milli / 10**6) == (/ 10 ** 9)
    print('%d x %d matmul took:   \t%.4f ms,\t %.2f GFLOPS' % (n, n, elapsed_ms, rate,), file=sys.stderr)
    return rate, elapsed_ms


def bench_v2(n: int):
    times = []
    with tf.device("/%s:0" % (_ARGS_DEVICE)):
        matrix1 = tf.Variable(tf.ones((n, n), dtype=_ARGS_DTYPE))
        matrix2 = tf.Variable(tf.ones((n, n), dtype=_ARGS_DTYPE))

        for i in range(_ARGS_REPS):
            start = time.monotonic()
            product = tf.matmul(matrix1, matrix2)
            times.append(time.monotonic() - start)

    times_ms = 1000 * np.array(times)  # in seconds, convert to ms
    elapsed_ms = np.median(times_ms)

    ops = n ** 3 + (n - 1) * n ** 2  # n^2*(n-1) additions, n^3 multiplications
    rate = ops / elapsed_ms / 10 ** 6  # in GFLOPS. (/ milli / 10**6) == (/ 10 ** 9)
    print('%d x %d matmul took:   \t%.4f ms,\t %.2f GFLOPS' % (n, n, elapsed_ms, rate,), file=sys.stderr)
    return rate, elapsed_ms


def main():
    np.set_printoptions(suppress=True)
    tf_version = tf.__version__
    print("# Version: %s, path: %s" % (tf_version, tf.__path__), file=sys.stderr)

    if int(tf_version[0]) >= 2:
        rate, elapsed = bench_v2(_ARGS_MATRIX_SIZE)
    else:
        rate, elapsed = bench_v1(_ARGS_MATRIX_SIZE)

    result = {
        "component": "tensorflow",
        "name": "PiMatmul",
        "@parameters": {
            "dtype": _ARGS_DTYPE,
            "device": _ARGS_DEVICE,
            "reps": _ARGS_REPS,
            "matrix_size": _ARGS_MATRIX_SIZE,
        },
        "@result": {
            "rate": rate,
            "elapsed": elapsed,
        },
        "tensorflow_buildinfo": _get_aicoe_tensorflow_build_info()
    }
    json.dump(result, sys.stdout, indent=2)


if __name__ == '__main__':
    main()
