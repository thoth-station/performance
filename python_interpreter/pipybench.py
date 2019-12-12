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

"""Performance Indicator (PI): PyBench (Thoth Team)."""

import sys
import json
import os
from pybench.pybench import PyBenchCmdline

_ARGS_ROUNDS = int(os.getenv("ROUNDS", 10))
print("ROUNDS set to %s" % _ARGS_ROUNDS, file=sys.stderr)

# pybench prints stats to stdout, we need to move them to stderr so that amun
# reports back just reported JSON.
original_stdout = sys.stdout
sys.stdout = os.fdopen(os.dup(sys.stderr.fileno()), sys.stdout.mode)
bench = PyBenchCmdline(
    argv=["pybench", "-n", str(_ARGS_ROUNDS), "-o", "json", "-f", "output.json"]
)

with open("output.json", "r") as output:
    benchmarks = json.load(output)

result = {
    "library": "pybench",
    "name": "PiPyBench",
    "@parameters": {
        "rounds": _ARGS_ROUNDS,
    },
    "@result": benchmarks,
}
json.dump(result, original_stdout, indent=2)
sys.exit(bench.exit_code)
