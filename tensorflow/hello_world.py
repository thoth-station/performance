#!/usr/bin/env python3
#
# Copyright 2020 Fridolin Pokorny <fridolin@redhat.com>
# Licensed under Apache-2
#
# This is not really a performance indicator, but can be used to check correct
# initialization of TF. We cannot rely on slow terminals, and slow output
# (unbuffered stderr).

import json
import logging
import os
import sys
from timeit import time
import tensorflow as tf

_LOGGER = logging.getLogger("thoth.pi.hello_world")


# Number of repetitions.
# Options:
#   A positive integer.
_ARGS_REPS = int(os.getenv("HELLO_WORLD_REPS", 1))
print("HELLO_WORLD_REPS set to %s" % _ARGS_REPS, file=sys.stderr)


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


def _get_tensorflow_build_info():
    """Get tensorflow build info provided by tensorflow 2.3 and above."""
    try:
        return tf.sysconfig.get_build_info()
    except AttributeError:
        return None


def bench_v1():
    """Use v1 API for printing hello world."""
    start = time.monotonic()
    for i in range(_ARGS_REPS):
        hello = tf.constant("Hello, TensorFlow by Thoth!")
        sess = tf.Session()
        print(sess.run(hello), file=sys.stderr)
        del hello
        del sess

    return time.monotonic() - start


def bench_v2():
    """Use v2 API for printing hello world."""
    start = time.monotonic()

    for i in range(_ARGS_REPS):
        hello = tf.constant("Hello, TensorFlow by Thoth!")
        tf.print(hello, output_stream=sys.stderr)
        del hello

    return time.monotonic() - start


def main():
    """Main entrypoint."""
    tf_version = tf.__version__
    print("# Version: %s, path: %s" % (tf_version, tf.__path__), file=sys.stderr)

    if int(tf_version.split(".", maxsplit=1)[0]) == 1:
        elapsed = bench_v1()
    else:
        elapsed = bench_v2()

    result = {
        "component": "tensorflow",
        "name": "PiHelloWorld",
        "@parameters": {"HELLO_WORLD_REPS": _ARGS_REPS,},
        "@result": {"elapsed": elapsed},
        "tensorflow_aicoe_buildinfo": _get_aicoe_tensorflow_build_info(),
        "tensorflow_upstream_buildinfo": _get_tensorflow_build_info(),
    }
    json.dump(result, sys.stdout, indent=2, sort_keys=True)


__name__ == "__main__" and main()
