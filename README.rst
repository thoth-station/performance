Performance Scripts
-------------------

This repository contains a set of scripts that can be used directly by Amun to
check performance of an application stack. This application stack can be
supplied directly by user (to verify its stack performance) or can be generated
by Dependency Monkey to gather software stack characteristics.

Usage
=====

Amun accepts an URL to the script (or you can submit it verbatim as a string)
to Amun API as part of the request (parameter ``script``).

To use script directly from GitHub, you can open the given script to view its
content and than click on "Raw" button in the GitHub's file header to obtain an
URL to a raw script file. Use the URL as the ``script`` parameter on Amun
API (do not pass directly "non-raw" URL as Amun will download HTML page instead
of raw file content).

Writing a performance script
============================


The performance script should be directly executable (e.g. ``python3
script.py``), it can print additional information onto ``stderr`` in any form
(this output is captured by Amun for additional analaysis). The output written
onto ``stdout`` *has to be* in a JSON format with any keys and values the
script wants to capture. To automatically submit results into Thoth's
knowledge base, there is expected key ``performance_index`` key which value has
to be from 0.0 to 1.0 inclusively. This number states how well the given
application performs in the performance script run.

Example local run
=================

.. code-block:: console

  # Clone this repo and cd into tensorflow directory:
  $ git clone https://github.com/thoth-station/performance.git
  $ cd performance/tensorflow
  # Install TensorFlow:
  $ pipenv install tensorflow==1.9.0 --python 3.6
  # Run the matmul.py performance:
  $ pipenv run python3 ./matmul.py
  DTYPE set to float32
  DEVICE set to cpu
  REPS set to 5
  # Version: 1.9.0, path: ['/home/fpokorny/.local/share/virtualenvs/tensorflow-bF0KqMxV/lib/python3.6/site-packages/tensorflow']

  size,time,flop

  256 x 256 matmul took:        0.8357 ms,       40.07 GFLOPS
  512 x 512 matmul took:        2.5985 ms,       103.20 GFLOPS
  1024 x 1024 matmul took:      8.7373 ms,       245.66 GFLOPS
  2048 x 2048 matmul took:      69.1140 ms,      248.51 GFLOPS
  4096 x 4096 matmul took:      526.8250 ms,     260.85 GFLOPS
  8192 x 8192 matmul took:      4641.6416 ms,    236.87 GFLOPS
  16384 x 16384 matmul took:    38691.7303 ms,   227.33 GFLOPS
  {
    "256": {
      "elapsed_ms": 0.8356571197509766,
      "rate": 40.07492452165022
    },
    "512": {
      "elapsed_ms": 2.5985240936279297,
      "rate": 103.20216489722434
    },
    "1024": {
      "elapsed_ms": 8.737325668334961,
      "rate": 245.6627065852563
    },
    "2048": {
      "elapsed_ms": 69.11396980285645,
      "rate": 248.5123474891199
    },
    "4096": {
      "elapsed_ms": 526.824951171875,
      "rate": 260.8497869174887
    },
    "8192": {
      "elapsed_ms": 4641.641616821289,
      "rate": 236.8654475450276
    },
    "16384": {
      "elapsed_ms": 38691.730260849,
      "rate": 227.33086702127227
    }
  }

Please note that the JSON output is printed to ``stdout``, other messages go to ``stderr``.
