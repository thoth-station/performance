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

"""Performance Indicator (PI): PyPerformance (Thoth Team)."""

import logging
import sys
import os
import json
import subprocess
from pathlib import Path

from pyperformance.cli import parse_args
from pyperformance.cli_run import get_benchmarks_to_run
from pyperformance.run import run_benchmarks

_LOGGER = logging.getLogger(__name__)


def bench():
    parser, options = parse_args()
    bench_funcs, bench_groups, should_run = get_benchmarks_to_run(options)
    cmd_prefix = [sys.executable]
    suite, errors = run_benchmarks(bench_funcs, should_run, cmd_prefix, options)
    suite.dump(options.output)

original_stdout = sys.stdout
sys.stdout = os.fdopen(os.dup(sys.stderr.fileno()), sys.stdout.mode)
sys.argv.extend(["run", "--python=python3", "-o", "output.json", "-b", "2n3"])
try:
    bench()
except SystemExit as exc:
    if int(str(exc)) != 0:
        print("pyperformance did not finish successfully: %d", int(exc), file=sys.stderr)

with open("output.json", "r") as output:
    benchmarks = json.load(output)

result = {
    "library": "performance",
    "name": "PiPyPerformance",
    "@parameters": {},
    "@result": benchmarks,
}

json.dump(result, original_stdout, indent=2)
sys.exit(0)

