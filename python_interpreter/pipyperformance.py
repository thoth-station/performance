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

"""Performance Indicator (PI): PyPerformance (Thoth Team)."""

import logging
import sys
import os
import json
import subprocess
from pathlib import Path

from pyperformance.cli import main

_LOGGER = logging.getLogger(__name__)


# pyperformance prints stats to stdout, we need to move them to stderr so that amun
# reports back just reported JSON.
original_stdout = sys.stdout
sys.stdout = os.fdopen(os.dup(sys.stderr.fileno()), sys.stdout.mode)
sys.argv.extend(["run", "--python=python3", "-o", "output.json"])
bench = main()

with open("output.json", "r") as output:
    benchmarks = json.load(output)

result = {
    "test_suite": "performance",
    "name": "PiPyPerformance",
    "@parameters": {},
    "@result": benchmarks,
}

json.dump(result, original_stdout, indent=2)
sys.exit(bench.exit_code)

