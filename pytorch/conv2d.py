#!/usr/bin/env python3
# Copyright(C) 2019 Francesco Murdaca
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

"""Performance Indicator (PI): Conv2D for PyTorch (Thoth Team)."""

import logging
import os
import sys
import numpy as np
import json
from timeit import time

import torch

_LOGGER = logging.getLogger(__name__)

_ARGS_DTYPE_MAP = {
    "float32": {"dtype": torch.float32},
    "float64": {"dtype": torch.float64},
    "float16": {"dtype": torch.float16},
    "uint8": {"dtype": torch.uint8},
    "int8": {"dtype": torch.int8},
    "int16": {"dtype": torch.int16},
    "int32": {"dtype": torch.int32},
    "int64": {"dtype": torch.int64},
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
_ARGS_DTYPE = os.getenv("MATMUL_DTYPE", "float32")

if _ARGS_DTYPE not in (
    "float32",
    "float64",
    "float16" "uint8",
    "int8",
    "int16",
    "int32",
    "int64",
):
    raise ValueError("Unknown MATMUL_DTYPE")
print("DTYPE set to %s" % _ARGS_DTYPE, file=sys.stderr)

# # Run on CPU or GPU.
# # Options:
# #   cpu
# #   gpu
_ARGS_DEVICE = os.getenv("CONV2D_DEVICE", "cpu")
print("DEVICE set to %s" % _ARGS_DEVICE, file=sys.stderr)

# Number of repetitions.
# Options:
#   A positive integer.
_ARGS_REPS = int(os.getenv("CONV_REPS", 80))
print("REPS set to %s" % _ARGS_REPS, file=sys.stderr)

# Data format
# # Options:
# #   NCHW Channel_first (Num_samples(N) x Channels(C) x Height(H) x Width(W))
_ARGS_DATA_FORMAT = os.getenv("CONV_DATA_FORMAT", "NCHW")
print("CONV DATA FORMAT set to %s" % _ARGS_DATA_FORMAT, file=sys.stderr)

# INPUT TENSOR
_ARGS_BATCH = int(os.getenv("BATCH", 4))  # Number of images per convolution
print("BATCH set to %s" % _ARGS_BATCH, file=sys.stderr)

_ARGS_INPUT_HEIGHT = int(os.getenv("TENSOR_INPUT_HEIGHT", 700))
print("TENSOR INPUT HEIGHT set to %s" % _ARGS_INPUT_HEIGHT, file=sys.stderr)

_ARGS_INPUT_WIDTH = int(os.getenv("TENSOR_INPUT_WIDTH", 161))
print("TENSOR INPUT WIDTH set to %s" % _ARGS_INPUT_WIDTH, file=sys.stderr)

_ARGS_INPUT_CHANNELS = int(os.getenv("TENSOR_INPUT_CHANNELS", 1))
print("TENSOR INPUT CHANNELS set to %s" % _ARGS_INPUT_CHANNELS, file=sys.stderr)

# FILTER
_ARGS_FILTER_HEIGHT = int(os.getenv("FILTER_INPUT_HEIGHT", 20))
print("FILTER INPUT HEIGHT set to %s" % _ARGS_FILTER_HEIGHT, file=sys.stderr)

_ARGS_FILTER_WIDTH = int(os.getenv("FILTER_INPUT_WIDTH", 5))
print("FILTER INPUT WIDTH set to %s" % _ARGS_FILTER_WIDTH, file=sys.stderr)

_ARGS_OUTPUT_CHANNELS = int(os.getenv("FILTER_OUTPUT_CHANNELS", 32))
print("FILTER OUTPUT CHANNELS set to %s" % _ARGS_OUTPUT_CHANNELS, file=sys.stderr)

# Padding
_ARGS_PADDING = int(os.getenv("FILTER_PADDING", 1))
print("FILTER PADDING set to %s" % _ARGS_PADDING, file=sys.stderr)

# Stride, the speed by which the filter moves across the image
_ARGS_STRIDES = int(os.getenv("FILTER_STRIDES", 2))
print("FILTER STRIDES set to %s" % _ARGS_STRIDES, file=sys.stderr)

if _ARGS_DEVICE == "cpu":
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def bench(
    batch: int,
    tensor_input_height: int,
    tensor_input_width: int,
    tensor_input_channels: int,
    tensor_output_channels: int,
    filter_height: int,
    filter_width: int,
):
    if _ARGS_DEVICE == "gpu":
        if torch.cuda.is_available():
            number_GPU = torch.cuda.device_count()
            current_device = torch.cuda.current_device()
            torch.cuda.device(current_device)
            name_GPU = torch.cuda.get_device_name(current_device)
            device = torch.device("cuda")
        else:
            raise Exception("No GPU available")
    else:
        device = torch.device("cpu")

    input_tensor = torch.ones(
        (batch, tensor_input_channels, tensor_input_height, tensor_input_width),
        dtype=_ARGS_DTYPE_MAP[_ARGS_DTYPE]["dtype"],
        device=device,
    )

    convolution = torch.nn.Conv2d(
        in_channels=tensor_input_channels,
        out_channels=tensor_output_channels,
        kernel_size=(filter_height, filter_width),
        stride=(_ARGS_STRIDES, _ARGS_STRIDES),
        padding=(_ARGS_PADDING, _ARGS_PADDING),
    )

    times = []

    for i in range(_ARGS_REPS):
        start = time.monotonic()
        convolution(input_tensor)
        times.append(time.monotonic() - start)

    times_ms = 1000 * np.array(times)  # in seconds, convert to ms
    elapsed_ms = np.median(times_ms)
    # Source:
    # https://github.com/tensorflow/tensorflow/blob/c19e29306ce1777456b2dbb3a14f511edf7883a8/tensorflow/python/profiler/internal/flops_registry.py#L381
    # Formula:
    #  batch_size * image_x_dim * image_y_dim * kernel_x_dim * kernel_y_dim
    #  * input_depth * output_depth * 2 / (image_x_stride * image_x_stride)
    ops = (
        batch
        * tensor_input_height
        * tensor_input_width
        * filter_height
        * filter_width
        * tensor_input_channels
        * tensor_output_channels
        * 2
    ) / (_ARGS_STRIDES * _ARGS_STRIDES)
    rate = ops / elapsed_ms / 10 ** 6  # in GFLOPS. (/ milli / 10**6) == (/ 10 ** 9)
    print("conv took:   \t%.4f ms,\t %.2f GFLOPS" % (elapsed_ms, rate), file=sys.stderr)

    return rate, elapsed_ms


def main():
    np.set_printoptions(suppress=True)
    print(
        "# Version: %s, path: %s" % (torch.__version__, torch.__path__), file=sys.stderr
    )

    rate, elapsed = bench(
        batch=_ARGS_BATCH,
        tensor_input_height=_ARGS_INPUT_HEIGHT,
        tensor_input_width=_ARGS_INPUT_WIDTH,
        tensor_input_channels=_ARGS_INPUT_CHANNELS,
        tensor_output_channels=_ARGS_OUTPUT_CHANNELS,
        filter_height=_ARGS_FILTER_HEIGHT,
        filter_width=_ARGS_FILTER_WIDTH,
    )

    result = {
        "component": "pytorch",
        "name": "PiConv2D",
        "@parameters": {
            "dtype": _ARGS_DTYPE,
            "device": _ARGS_DEVICE,
            "reps": _ARGS_REPS,
            "batch": _ARGS_BATCH,
            "input_height": _ARGS_INPUT_HEIGHT,
            "input_width": _ARGS_INPUT_WIDTH,
            "input_channels": _ARGS_INPUT_CHANNELS,
            "filter_height": _ARGS_FILTER_HEIGHT,
            "filter_width": _ARGS_FILTER_WIDTH,
            "output_channels": _ARGS_OUTPUT_CHANNELS,
            "strides": _ARGS_STRIDES,
            "padding": _ARGS_PADDING,
            "data_format": _ARGS_DATA_FORMAT,
        },
        "@result": {"rate": rate, "elapsed": elapsed},
        "pytorch_buildinfo": None,
    }
    json.dump(result, sys.stdout, indent=2)


if __name__ == "__main__":
    main()
