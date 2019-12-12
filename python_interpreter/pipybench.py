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
        "rounds": _ARGS_ROUNDS
    },
    "@result": {
        "built_in_function_calls_average": benchmarks["results"]["BuiltinFunctionCalls"]["average"],
        "built_in_method_lookup_average": benchmarks["results"]["BuiltinMethodLookup"]["average"],
        "compare_floats_average": benchmarks["results"]["CompareFloats"]["average"],
        "compare_floats_integers_average": benchmarks["results"]["CompareFloatsIntegers"]["average"],
        "compare_integers_average": benchmarks["results"]["CompareIntegers"]["average"],
        "compare_interned_strings_average": benchmarks["results"]["CompareInternedStrings"]["average"],
        "compare_longs_average": benchmarks["results"]["CompareLongs"]["average"],
        "compare_strings_average": benchmarks["results"]["CompareStrings"]["average"],
        "compare_unicode_average": benchmarks["results"]["CompareUnicode"]["average"],
        "concat_strings_average": benchmarks["results"]["ConcatStrings"]["average"],
        "concat_unicode_average": benchmarks["results"]["ConcatUnicode"]["average"],
        "create_instances_average": benchmarks["results"]["CreateInstances"]["average"],
        "create_new_instances_average": benchmarks["results"]["CreateNewInstances"]["average"],
        "create_strings_with_concat_average": benchmarks["results"]["CreateStringsWithConcat"]["average"],
        "create_unicode_with_concat_average": benchmarks["results"]["CreateUnicodeWithConcat"]["average"],
        "dict_creation_average": benchmarks["results"]["DictCreation"]["average"],
        "dict_with_float_keys_average": benchmarks["results"]["DictWithFloatKeys"]["average"],
        "dict_with_integer_keys_average": benchmarks["results"]["DictWithIntegerKeys"]["average"],
        "dict_with_string_keys_average": benchmarks["results"]["DictWithStringKeys"]["average"],
        "for_loops_average": benchmarks["results"]["ForLoops"]["average"],
        "if_then_else_average": benchmarks["results"]["IfThenElse"]["average"],
        "list_slicing_average": benchmarks["results"]["ListSlicing"]["average"],
        "nested_for_loops_average": benchmarks["results"]["NestedForLoops"]["average"],
        "normal_class_attribute_average": benchmarks["results"]["NormalClassAttribute"]["average"],
        "normal_instance_attribute_average": benchmarks["results"]["NormalInstanceAttribute"]["average"],
        "python_function_calls_average": benchmarks["results"]["PythonFunctionCalls"]["average"],
        "python_method_calls_average": benchmarks["results"]["PythonMethodCalls"]["average"],
        "recursion_average": benchmarks["results"]["Recursion"]["average"],
        "second_import_average": benchmarks["results"]["SecondImport"]["average"],
        "second_package_import_average": benchmarks["results"]["SecondPackageImport"]["average"],
        "second_submodule_import_average": benchmarks["results"]["SecondSubmoduleImport"]["average"],
        "simple_complex_arithmetic_average": benchmarks["results"]["SimpleComplexArithmetic"]["average"],
        "simple_dict_manipulation_average": benchmarks["results"]["SimpleDictManipulation"]["average"],
        "simple_float_arithmetic_average": benchmarks["results"]["SimpleFloatArithmetic"]["average"],
        "simple_int_float_arithmetic_average": benchmarks["results"]["SimpleIntFloatArithmetic"]["average"],
        "simple_integer_arithmetic_average": benchmarks["results"]["SimpleIntegerArithmetic"]["average"],
        "simple_list_manipulation_average": benchmarks["results"]["SimpleListManipulation"]["average"],
        "simple_long_arithmetic_average": benchmarks["results"]["SimpleLongArithmetic"]["average"],
        "small_lists_average": benchmarks["results"]["SmallLists"]["average"],
        "small_tuples_average": benchmarks["results"]["SmallTuples"]["average"],
        "special_class_attribute_average": benchmarks["results"]["SpecialClassAttribute"]["average"],
        "special_instance_attribute_average": benchmarks["results"]["SpecialInstanceAttribute"]["average"],
        "string_mappings_average": benchmarks["results"]["StringMappings"]["average"],
        "string_predicates_average": benchmarks["results"]["StringPredicates"]["average"],
        "string_slicing_average": benchmarks["results"]["StringSlicing"]["average"],
        "try_except_average": benchmarks["results"]["TryExcept"]["average"],
        "try_raise_except_average": benchmarks["results"]["TryRaiseExcept"]["average"],
        "tuple_slicing_average": benchmarks["results"]["TupleSlicing"]["average"],
        "unicode_mappings_average": benchmarks["results"]["UnicodeMappings"]["average"],
        "unicode_predicates_average": benchmarks["results"]["UnicodePredicates"]["average"],
        "unicode_properties_average": benchmarks["results"]["UnicodeProperties"]["average"],
        "unicode_slicing_average": benchmarks["results"]["UnicodeSlicing"]["average"],
        "totals_average": benchmarks["results"]["Totals"]["average"],
    }
}
json.dump(result, original_stdout, indent=2)
sys.exit(bench.exit_code)
