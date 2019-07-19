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
(this output is captured by Amun for additional analysis). The output written
onto ``stdout`` *has to be* in a JSON format with any keys and values the
script wants to capture. To automatically submit results into Thoth's
knowledge base, there is expected key ``overall_score`` key which value has
to be from 0.0 to 1.0 inclusively. This number states how well the given
application performs in the performance script run.

The script *has to report* following information to stdout in a form of JSON which states following:

* `@parameters` - parameters which define the given performance script (e.g. metrix size in case of matrix multiplication)
* `@result` - the actual result which was obtained during the performance indicator run

Example:

.. code-block:: json

  {
    "@parameters": {
      "dtype": "float32",
      "device": "cpu",
      "reps": 20000,
      "matrix_size": 512
    },
    "@result": {
      "rate": 0.009799366109955314,
      "elapsed": 27366.39380455017
    }
 }

The related model on the graph database side should state `rate`, `elapsed`,
`dtype`, `device`, `reps` and `matrix_size` properties to capture all the
relevant info of the performance indicator run. This is done automatically by
syncing logic once the `model is created
<https://github.com/thoth-station/storages#creating-own-performance-indicators>`_
and `graph schema is updated
<https://github.com/thoth-station/storages#schema-adjustment-in-deployment>`_.

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
  REPS set to 20000
  MATRIX size set to 512
  # Version: 1.13.1, path: ['/home/fpokorny/.local/share/virtualenvs/tensorflow-FigIdZQa/lib/python3.6/site-packages/tensorflow_estimator/python/estimator/api', '/home/fpokorny/.local/share/virtualenvs/tensorflow-FigIdZQa/lib/python3.6/site-packages/tensorflow', '/home/fpokorny/.local/share/virtualenvs/tensorflow-FigIdZQa/lib/python3.6/site-packages/tensorflow/_api/v1']
  WARNING:tensorflow:From /home/fpokorny/.local/share/virtualenvs/tensorflow-FigIdZQa/lib/python3.6/site-packages/tensorflow/python/framework/op_def_library.py:263: colocate_with (from tensorflow.python.framework.ops) is deprecated and will be removed in a future version.
  Instructions for updating:
  Colocations handled automatically by placer.
  512 x 512 matmul took:   	1.2320 ms,	 217.67 GFLOPS
  {
    "framework": "tensorflow",
    "name": "PiMatmul",
    "@parameters": {
      "dtype": "float32",
      "device": "cpu",
      "reps": 20000,
      "matrix_size": 512
    },
    "@result": {
      "rate": 0.006877627136404815,
      "elapsed": 38992.12718009949
    },
    "tensorflow_buildinfo": {
      "source_HEAD": "6612da89516247503f03ef76e974b51a434fb52e",
      "source_remote_origin": "https://github.com/tensorflow/tensorflow.git",
      "OS_VER": "Fedora release 28 (Twenty Eight)",
      "GLIBC_VER": "ldd (GNU libc) 2.27",
      "PIP_VER": "pip 19.0.3 from /usr/local/lib/python3.6/site-packages/pip (python 3.6)",
      "PROTOC_VER": "libprotoc 3.5.0",
      "LOGICAL_CPUS": "64",
      "CORES_PER_PCPU": " 1",
      "PHYSICAL_CPUS": "64",
      "GCC_VER": "gcc (GCC) 8.2.1 20181215 (Red Hat 8.2.1-6)",
      "OS": "Linux",
      "kernel": "3.10.0-862.9.1.el7.x86_64",
      "architecture": "skylake",
      "processor": "Intel Core Processor (Skylake, IBRS)",
      "Bazel_version": "Build label: 0.20.0",
      "Java_version": "1.8.0_201",
      "Python_version": "3.6.8",
      "gpp_version": "g++ (GCC) 8.2.1 20181215 (Red Hat 8.2.1-6)",
      "swig_version": "",
      "NVIDIA_driver_version": "",
      "CUDA_device_count": "0",
      "CUDA_device_names": "",
      "CUDA_toolkit_version": "",
      "GCC_FLAGS": "-march=skylake -mmmx -msse -msse2 -msse3 -mssse3 -mcx16 -msahf -mmovbe -maes -mpclmul -mpopcnt -mabm -mfma -mbmi -mbmi2 -mavx -mavx2 -msse4.2 -msse4.1 -mlzcnt -mrtm -mhle -mrdrnd -mf16c -mfsgsbase -mrdseed -mprfchw -madx -mfxsr -mxsave -mxsaveopt -mavx512f -mavx512cd -mclflushopt -mxsavec -mavx512dq -mavx512bw -mavx512vl -mpku --param l1-cache-size=32 --param l1-cache-line-size=64 --param l2-cache-size=16384 -mtune=skylake",
      "CPUINFO_FLAGS": " fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ss syscall nx pdpe1gb rdtscp lm constant_tsc rep_good nopl xtopology eagerfpu pni pclmulqdq vmx ssse3 fma cx16 pcid sse4_1 sse4_2 x2apic movbe popcnt tsc_deadline_timer aes xsave avx f16c rdrand hypervisor lahf_lm abm 3dnowprefetch ibrs ibpb stibp tpr_shadow vnmi flexpriority ept vpid fsgsbase tsc_adjust bmi1 hle avx2 smep bmi2 erms invpcid rtm mpx avx512f avx512dq rdseed adx smap clflushopt avx512cd avx512bw avx512vl xsaveopt xsavec xgetbv1 arat pku ospke spec_ctrl intel_stibp",
      "CPUINFO_FLAGS_TENSORFLOW": "sse sse2 ssse3 fma sse4_1 sse4_2 avx avx2 avx512f avx512dq avx512cd avx512bw avx512vl ",
      "CPU_FAMILY": "6",
      "CPU_MODEL": "94",
      "GCC_HOST_COMPILER_PATH": "/usr/bin/gcc",
      "CUDA_TOOLKIT_PATH": "/usr/local/cuda",
      "CUDNN_INSTALL_PATH": "/usr/local/cuda",
      "JAVA_HOME": "/usr/lib/jvm/java-1.8.0-openjdk-1.8.0.162-3.b12.fc28.x86_64 /usr/lib/jvm/java-1.8.0-openjdk-1.8.0.201.b09-2.fc28.x86_64",
      "PYTHON_LIB_PATH": "/usr/lib64/python3.6/site-packages",
      "LD_LIBRARY_PATH": "/usr/lib64:/usr/local/lib:/usr/local/lib;",
      "PYTHON_BIN_PATH": "/usr/bin/python3.6",
      "PATH": "/home/default/bin:/usr/local/bin:/opt/app-root/src/bin:/opt/app-root/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/home/default/.local/bin",
      "PORT": "",
      "BUILD_OPTS": "",
      "PYTHON_VERSION": "3.6",
      "HOST_ON_HTTP_SERVER": "n",
      "TEST_WHEEL_FILE": "y",
      "GIT_RELEASE_REPO": "https://github.com/AICoE/tensorflow-wheels.git",
      "CUSTOM_BUILD": "bazel build --copt=-mavx --copt=-mavx2 --copt=-mfma --copt=-mfpmath=both --copt=-msse4.2 --cxxopt='-D_GLIBCXX_USE_CXX11_ABI=0' --local_resources 2048,2.0,1.0 --verbose_failures //tensorflow/tools/pip_package:build_pip_package",
      "TF_NEED_TENSORRT": "0",
      "TF_ENABLE_XLA": "0",
      "TF_NEED_VERBS": "0",
      "TF_NEED_S3": "0",
      "TF_CUDA_VERSION": "9.2",
      "TF_CUDA_COMPUTE_CAPABILITIES": "3.0,3.5,5.2,6.0,6.1,7.0",
      "TF_NEED_HDFS": "0",
      "TF_NEED_IGNITE": "0",
      "TF_NEED_GDR": "0",
      "TF_ENABLE_TEST": "0",
      "TF_DOWNLOAD_CLANG": "0",
      "TF_NEED_GCP": "0",
      "TF_CUDNN_VERSION": "7",
      "TF_NEED_AWS": "0",
      "TF_NEED_ROCM": "0",
      "TF_SET_ANDROID_WORKSPACE": "0",
      "TF_NEED_OPENCL": "0",
      "TF_GIT_BRANCH": "r1.13",
      "TF_CUDA_CLANG": "0",
      "TF_NEED_JEMALLOC": "1",
      "TF_NEED_KAFKA": "0",
      "TF_NEED_MPI": "0",
      "TF_NEED_CUDA": "0",
      "TF_NEED_OPENCL_SYCL": "0",
      "march": "skylake"
    }
  }

Please note that the JSON output is printed to ``stdout``, other messages go to
``stderr``. Key `tensorflow_buildinfo` is reported by the script, but is not
part of the actual `@result`. TensorFlow's build information is parsed from
custom AICoE TensorFlow builds present on `AICoE experimental index
<http://tensorflow.pypi.thoth-station.ninja>`_.


