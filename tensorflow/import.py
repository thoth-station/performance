#!/usr/bin/env python3
#
# Copyright 2020 Fridolin Pokorny <fridolin@redhat.com>
# Licensed under Apache-2
#
# It looks like the very first import takes some time, the remaining ones (e.g.
# triggered using importlib.reload) are not that expensive. Hence, no
# repetitions parametrized.

import json
import logging
import os
import sys
from timeit import time

_LOGGER = logging.getLogger("thoth.pi.import")


def _get_aicoe_tensorflow_build_info(tf):
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


def _get_tensorflow_build_info(tf):
    """Get tensorflow build info provided by tensorflow 2.3 and above."""
    try:
        return tf.sysconfig.get_build_info()
    except AttributeError:
        return None


def main():
    """Main entrypoint."""
    start = time.monotonic()
    import tensorflow as tf

    end = time.monotonic()

    tf_version = tf.__version__
    print("# Version: %s, path: %s" % (tf_version, tf.__path__), file=sys.stderr)

    result = {
        "component": "tensorflow",
        "name": "PiImport",
        "@parameters": {},
        "@result": {"elapsed": end - start,},
        "tensorflow_aicoe_buildinfo": _get_aicoe_tensorflow_build_info(tf),
        "tensorflow_upstream_buildinfo": _get_tensorflow_build_info(tf),
    }
    json.dump(result, sys.stdout, indent=2, sort_keys=True)


__name__ == "__main__" and main()
