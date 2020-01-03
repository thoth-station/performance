#!/usr/bin/env python3
# Copyright(C) 2019, 2020 Francesco Murdaca
#
# This program is free software: you can redistribute it and / or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

"""Performance Indicator (PI): matmul for PyTorch (Thoth Team)."""

import logging
import os
import sys
import numpy as np
import json
from timeit import time

import torch

_LOGGER = logging.getLogger(__name__)

_ARGS_DTYPE_MAP = {
    "float32": {
        "dtype": torch.float32,
    },
    "float64": {
        "dtype": torch.float64,
    },
    "float16": {
        "dtype": torch.float16,
    },
    "uint8": {
        "dtype": torch.uint8,
    },
    "int8": {
        "dtype": torch.int8,
    },
    "int16": {
        "dtype": torch.int16,
    },
    "int32": {
        "dtype": torch.int32,
    },
    "int64": {
        "dtype": torch.int64,
    },
}

# Datatype used.
# Options:
#   32-bit floating point == float32
#   64-bit floating point == float64
#   16-bit floating point == float16
#   8-bit integer (unsigned) == uint8
#   8-bit integer (signed) == int8
#   16-bit integer (signed) == int16
#   32-bit integer (signed) == int32
#   64-bit integer (signed) == int64
_ARGS_DTYPE = os.getenv('MATMUL_DTYPE', 'float32')

if _ARGS_DTYPE not in (
    'float32',
    'float64', 
    'float16'
    'uint8',
    'int8',
    'int16',
    'int32',
    'int64',
  ):
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
_ARGS_MATRIX_SIZE = int(os.getenv('MATMUL_MATRIX_SIZE', 64))
print("MATRIX size set to %s" % _ARGS_MATRIX_SIZE, file=sys.stderr)

if _ARGS_DEVICE == 'cpu':
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def bench(n):
    if _ARGS_DEVICE == 'gpu':
        if torch.cuda.is_available():
            number_GPU = torch.cuda.device_count()
            current_device = torch.cuda.current_device()
            torch.cuda.device(current_device)
            name_GPU = torch.cuda.get_device_name(current_device)
            device = torch.device('cuda')
        else:
            raise Exception("No GPU available")
    else:
        device = torch.device('cpu')

    matrix1 = torch.ones((n, n), dtype=_ARGS_DTYPE_MAP[_ARGS_DTYPE]["dtype"], device=device)
    matrix2 = torch.ones((n, n), dtype=_ARGS_DTYPE_MAP[_ARGS_DTYPE]["dtype"], device=device)

    times = []

    for i in range(_ARGS_REPS):
        start = time.monotonic()
        product = torch.mm(matrix1, matrix2)
        times.append(time.monotonic() - start)

    times_ms = 1000 * np.array(times)  # in seconds, convert to ms
    elapsed_ms = np.median(times_ms)

    ops = n ** 3 + (n - 1) * n ** 2  # n^2*(n-1) additions, n^3 multiplications
    rate = ops / elapsed_ms / 10 ** 6  # in GFLOPS. (/ milli / 10**6) == (/ 10 ** 9)
    print('%d x %d matmul took:   \t%.4f ms,\t %.2f GFLOPS' % (n, n, elapsed_ms, rate,), file=sys.stderr)
    return rate, elapsed_ms


def main():
    np.set_printoptions(suppress=True)
    print("# Version: %s, path: %s" % (torch.__version__, torch.__path__), file=sys.stderr)

    rate, elapsed = bench(_ARGS_MATRIX_SIZE)

    result = {
        "library": "pytorch",
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
        "pytorch_buildinfo": None
    }
    json.dump(result, sys.stdout, indent=2)


if __name__ == '__main__':
    main()
