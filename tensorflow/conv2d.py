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

"""Performance Indicator (PI): Conv2D for Tensorflow (Thoth Team)."""

import logging
import os
import sys
import numpy as np
import json
from timeit import time
import tensorflow as tf

_LOGGER = logging.getLogger(__name__)

# Datatype used.
# Options:
#   half
#   bfloat16
#   float32
#   float64
_ARGS_DTYPE = os.getenv("TENSOR_DTYPE", "float32")
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
# #   NHWC Channel_last (Num_samples(N) x Height(H) x Width(W) x Channels(C))
# #   NCHW Channel_first (Num_samples(N) x Channels(C) x Height(H) x Width(W))
_ARGS_DATA_FORMAT = os.getenv("CONV_DATA_FORMAT", "NHWC")
print("CONV DATA FORMAT set to %s" % _ARGS_DATA_FORMAT, file=sys.stderr)

# INPUT TENSOR
_ARGS_BATCH = int(os.getenv("BATCH", 4))  # Number of images per convolution
print("BATCH set to %s" % _ARGS_BATCH, file=sys.stderr)

_ARGS_INPUT_HEIGHT = int(os.getenv("TENSOR_INPUT_HEIGHT", 700))
print("TENSOR INPUT HEIGHT set to %s" % _ARGS_INPUT_HEIGHT, file=sys.stderr)

_ARGS_INPUT_WIDTH = int(os.getenv("TENSOR_INPUT_WIDTH", 161))
print("TENSOR INPUT WIDTH set to %s" % _ARGS_INPUT_WIDTH, file=sys.stderr)

_ARGS_T_INPUT_CHANNELS = int(os.getenv("TENSOR_INPUT_CHANNELS", 1))
print("TENSOR INPUT CHANNELS set to %s" % _ARGS_T_INPUT_CHANNELS, file=sys.stderr)

# FILTER
_ARGS_FILTER_HEIGHT = int(os.getenv("FILTER_INPUT_HEIGHT", 20))
print("FILTER INPUT HEIGHT set to %s" % _ARGS_FILTER_HEIGHT, file=sys.stderr)

_ARGS_FILTER_WIDTH = int(os.getenv("FILTER_INPUT_WIDTH", 5))
print("FILTER INPUT WIDTH set to %s" % _ARGS_FILTER_WIDTH, file=sys.stderr)

_ARGS_F_INPUT_CHANNELS = int(os.getenv("FILTER_INPUT_CHANNELS", _ARGS_T_INPUT_CHANNELS))
print("FILTER INPUT CHANNELS set to %s" % _ARGS_F_INPUT_CHANNELS, file=sys.stderr)

_ARGS_OUTPUT_CHANNELS = int(os.getenv("FILTER_OUTPUT_CHANNELS", 32))
print("FILTER OUTPUT CHANNELS set to %s" % _ARGS_OUTPUT_CHANNELS, file=sys.stderr)

# Padding
_ARGS_PADDING = os.getenv("FILTER_PADDING", "SAME")
print("FILTER PADDING set to %s" % _ARGS_PADDING, file=sys.stderr)

# Stride, the speed by which the filter moves across the image
_ARGS_STRIDES = int(os.getenv("FILTER_STRIDES", 2))
print("FILTER STRIDES set to %s" % _ARGS_STRIDES, file=sys.stderr)

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# if _ARGS_DEVICE == 'cpu':
#     os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def _get_aicoe_tensorflow_build_info():
    """Try to obtain information of AICoE TensorFlow builds.

    Do whatever is needed in this function, if there is an error, the reported build information is
    set to None (e.g. AICoE TensorFlow is not installed and such).
    """
    try:
        path = os.path.dirname(os.path.dirname(tf.__file__))
        build_info_path = os.path.join(
            path, "tensorflow-" + tf.__version__ + ".dist-info", "build_info.json"
        )
        with open(build_info_path, "r") as build_info_file:
            build_info = json.load(build_info_file)
        return build_info
    except Exception:
        _LOGGER.exception(
            "Failed to obtain AICoE specific build information for TensorFlow"
        )

    return None


def bench(
    batch: int,
    tensor_input_height: int,
    tensor_input_width: int,
    tensor_input_channels: int,
    filter_height: int,
    filter_width: int,
    filter_input_channels: int,
    filter_output_channels: int,
):
    g = tf.Graph()
    with tf.device("/%s:0" % (_ARGS_DEVICE)) and g.as_default():
        if _ARGS_DATA_FORMAT == "NHWC":
            init_tensor = tf.Variable(
                tf.ones(
                    [
                        batch,
                        tensor_input_height,
                        tensor_input_width,
                        tensor_input_channels,
                    ]
                ),
                dtype=_ARGS_DTYPE,
            )
            stride = [1] + [_ARGS_STRIDES, _ARGS_STRIDES] + [1]
        elif _ARGS_DATA_FORMAT == "NCHW":
            init_tensor = tf.Variable(
                tf.ones(
                    [
                        batch,
                        tensor_input_channels,
                        tensor_input_height,
                        tensor_input_width,
                    ]
                ),
                dtype=_ARGS_DTYPE,
            )
            stride = [1, 1] + [_ARGS_STRIDES, _ARGS_STRIDES]
        else:
            raise ValueError("Unknown data_format: " + str(_ARGS_DATA_FORMAT))
     
        init_filter = tf.Variable(
            tf.ones(
                [
                    filter_height,
                    filter_width,
                    filter_input_channels,
                    filter_output_channels,
                ]
            ),
            dtype=_ARGS_DTYPE,
        )
        convolution = tf.nn.conv2d(
            init_tensor,
            filter=init_filter,
            strides=stride,
            padding=_ARGS_PADDING,
            data_format=_ARGS_DATA_FORMAT,
        )

    times = []
    config = tf.ConfigProto()
    with tf.Session(graph=g, config=config) as sess:
        sess.run(tf.global_variables_initializer())
        # warmup
        sess.run(convolution.op)

        for i in range(_ARGS_REPS):
            start = time.monotonic()
            sess.run(convolution.op)
            times.append(time.monotonic() - start)

    times_ms = 1000 * np.array(times)  # in seconds, convert to ms
    elapsed_ms = np.median(times_ms)
    # Source: https://github.com/tensorflow/tensorflow/blob/c19e29306ce1777456b2dbb3a14f511edf7883a8/tensorflow/python/profiler/internal/flops_registry.py#L381
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
        * filter_output_channels
        * 2
    ) / (_ARGS_STRIDES * _ARGS_STRIDES)
    rate = ops / elapsed_ms / 10 ** 6  # in GFLOPS. (/ milli / 10**6) == (/ 10 ** 9)
    print('conv took:   \t%.4f ms,\t %.2f GFLOPS' % (elapsed_ms, rate), file=sys.stderr)

    return rate, elapsed_ms


def main():
    np.set_printoptions(suppress=True)
    print("# Version: %s, path: %s" % (tf.__version__, tf.__path__), file=sys.stderr)

    rate, elapsed = bench(
        batch=_ARGS_BATCH,
        tensor_input_height=_ARGS_INPUT_HEIGHT,
        tensor_input_width=_ARGS_INPUT_WIDTH,
        tensor_input_channels=_ARGS_T_INPUT_CHANNELS,
        filter_height=_ARGS_FILTER_HEIGHT,
        filter_width=_ARGS_FILTER_WIDTH,
        filter_input_channels=_ARGS_F_INPUT_CHANNELS,
        filter_output_channels=_ARGS_OUTPUT_CHANNELS
    )

    result = {
        "component": "tensorflow",
        "name": "PiConv2D",
        "@parameters": {
            "dtype": _ARGS_DTYPE,
            "device": _ARGS_DEVICE,
            "reps": _ARGS_REPS,
            "batch": _ARGS_BATCH,
            "input_height": _ARGS_INPUT_HEIGHT,
            "input_width": _ARGS_INPUT_WIDTH,
            "input_channels": _ARGS_T_INPUT_CHANNELS,
            "filter_height": _ARGS_FILTER_HEIGHT,
            "filter_width": _ARGS_FILTER_WIDTH,
            "output_channels": _ARGS_OUTPUT_CHANNELS,
            "strides": _ARGS_STRIDES,
            "padding": _ARGS_PADDING,
            "data_format": _ARGS_DATA_FORMAT,
        },
        "@result": {"rate": rate, "elapsed": elapsed},
        "tensorflow_buildinfo": _get_aicoe_tensorflow_build_info(),
    }
    json.dump(result, sys.stdout, indent=2)


if __name__ == "__main__":
    main()
