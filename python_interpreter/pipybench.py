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
    "test_suite": "pybench",
    "name": "PiPyBench",
    "@parameters": {
        "rounds": _ARGS_ROUNDS
    },
    "@result": {
        "BuiltinFunctionCalls_average": benchmarks["results"]["BuiltinFunctionCalls"]["average"],
        "BuiltinMethodLookup_average": benchmarks["results"]["BuiltinMethodLookup"]["average"],
        "CompareFloats_average": benchmarks["results"]["CompareFloats"]["average"],
        "CompareFloatsIntegers_average": benchmarks["results"]["CompareFloatsIntegers"]["average"],
        "CompareIntegers_average": benchmarks["results"]["CompareIntegers"]["average"],
        "CompareInternedStrings_average": benchmarks["results"]["CompareInternedStrings"]["average"],
        "CompareLongs_average": benchmarks["results"]["CompareLongs"]["average"],
        "CompareStrings_average": benchmarks["results"]["CompareStrings"]["average"],
        "CompareUnicode_average": benchmarks["results"]["CompareUnicode"]["average"],
        "ConcatStrings_average": benchmarks["results"]["ConcatStrings"]["average"],
        "ConcatUnicode_average": benchmarks["results"]["ConcatUnicode"]["average"],
        "CreateInstances_average": benchmarks["results"]["CreateInstances"]["average"],
        "CreateNewInstances_average": benchmarks["results"]["CreateNewInstances"]["average"],
        "CreateStringsWithConcat_average": benchmarks["results"]["CreateStringsWithConcat"]["average"],
        "CreateUnicodeWithConcat_average": benchmarks["results"]["CreateUnicodeWithConcat"]["average"],
        "DictCreation_average": benchmarks["results"]["DictCreation"]["average"],
        "DictWithFloatKeys_average": benchmarks["results"]["DictWithFloatKeys"]["average"],
        "DictWithIntegerKeys_average": benchmarks["results"]["DictWithIntegerKeys"]["average"],
        "DictWithStringKeys_average": benchmarks["results"]["DictWithStringKeys"]["average"],
        "ForLoops_average": benchmarks["results"]["ForLoops"]["average"],
        "IfThenElse_average": benchmarks["results"]["IfThenElse"]["average"],
        "ListSlicing_average": benchmarks["results"]["ListSlicing"]["average"],
        "NestedForLoops_average": benchmarks["results"]["NestedForLoops"]["average"],
        "NormalClassAttribute_average": benchmarks["results"]["NormalClassAttribute"]["average"],
        "NormalInstanceAttribute_average": benchmarks["results"]["NormalInstanceAttribute"]["average"],
        "PythonFunctionCalls_average": benchmarks["results"]["PythonFunctionCalls"]["average"],
        "PythonMethodCalls_average": benchmarks["results"]["PythonMethodCalls"]["average"],
        "Recursion_average": benchmarks["results"]["Recursion"]["average"],
        "SecondImport_average": benchmarks["results"]["SecondImport"]["average"],
        "SecondPackageImport_average": benchmarks["results"]["SecondPackageImport"]["average"],
        "SecondSubmoduleImport_average": benchmarks["results"]["SecondSubmoduleImport"]["average"],
        "SimpleComplexArithmetic_average": benchmarks["results"]["SimpleComplexArithmetic"]["average"],
        "SimpleDictManipulation_average": benchmarks["results"]["SimpleDictManipulation"]["average"],
        "SimpleFloatArithmetic_average": benchmarks["results"]["SimpleFloatArithmetic"]["average"],
        "SimpleIntFloatArithmetic_average": benchmarks["results"]["SimpleIntFloatArithmetic"]["average"],
        "SimpleIntegerArithmetic_average": benchmarks["results"]["SimpleIntegerArithmetic"]["average"],
        "SimpleListManipulation_average": benchmarks["results"]["SimpleListManipulation"]["average"],
        "SimpleLongArithmetic_average": benchmarks["results"]["SimpleLongArithmetic"]["average"],
        "SmallLists_average": benchmarks["results"]["SmallLists"]["average"],
        "SmallTuples_average": benchmarks["results"]["SmallTuples"]["average"],
        "SpecialClassAttribute_average": benchmarks["results"]["SpecialClassAttribute"]["average"],
        "SpecialInstanceAttribute_average": benchmarks["results"]["SpecialInstanceAttribute"]["average"],
        "StringMappings_average": benchmarks["results"]["StringMappings"]["average"],
        "StringPredicates_average": benchmarks["results"]["StringPredicates"]["average"],
        "StringSlicing_average": benchmarks["results"]["StringSlicing"]["average"],
        "TryExcept_average": benchmarks["results"]["TryExcept"]["average"],
        "TryRaiseExcept_average": benchmarks["results"]["TryRaiseExcept"]["average"],
        "TupleSlicing_average": benchmarks["results"]["TupleSlicing"]["average"],
        "UnicodeMappings_average": benchmarks["results"]["UnicodeMappings"]["average"],
        "UnicodePredicates_average": benchmarks["results"]["UnicodePredicates"]["average"],
        "UnicodeProperties_average": benchmarks["results"]["UnicodeProperties"]["average"],
        "UnicodeSlicing_average": benchmarks["results"]["UnicodeSlicing"]["average"],
        "Totals_average": benchmarks["results"]["Totals"]["average"],
    }
}
json.dump(result, original_stdout, indent=2)
sys.exit(bench.exit_code)
