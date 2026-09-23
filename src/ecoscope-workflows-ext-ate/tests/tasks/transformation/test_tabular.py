"""Tests for ecoscope_workflows_ext_ate.tasks.transformation._tabular.

Both functions here are registered via `wt_registry.register()`, a no-op
decorator at call time, so each is called directly as plain Python against
small, hand-built DataFrames.
"""

from __future__ import annotations

import pandas as pd
import pytest

from ecoscope_workflows_ext_ate.tasks.transformation._tabular import (
    apply_arithmetic_operation_over_columns,
    round_column_values,
)


class TestRoundColumnValues:
    def test_rounds_to_given_decimals(self):
        df = pd.DataFrame({"x": [1.234, 2.567]})

        result = round_column_values(df, column="x", output_column="x_rounded", decimals=1)

        assert result["x_rounded"].tolist() == [1.2, 2.6]

    def test_default_decimals_rounds_to_whole_numbers(self):
        df = pd.DataFrame({"x": [1.4, 1.5, 2.5]})

        result = round_column_values(df, column="x", output_column="x_rounded")

        # pandas uses banker's rounding: 1.5 -> 2, 2.5 -> 2.
        assert result["x_rounded"].tolist() == [1.0, 2.0, 2.0]

    def test_output_column_same_as_input_overwrites_in_place(self):
        df = pd.DataFrame({"x": [1.239]})

        result = round_column_values(df, column="x", output_column="x", decimals=2)

        assert result["x"].tolist() == [1.24]

    def test_original_column_preserved_when_output_column_differs(self):
        df = pd.DataFrame({"x": [1.239]})

        result = round_column_values(df, column="x", output_column="y", decimals=1)

        assert result["x"].tolist() == [1.239]
        assert result["y"].tolist() == [1.2]

    def test_missing_column_raises_key_error(self):
        df = pd.DataFrame({"x": [1.0]})

        with pytest.raises(KeyError):
            round_column_values(df, column="does_not_exist", output_column="y")


class TestApplyArithmeticOperationOverColumns:
    @pytest.mark.parametrize(
        "operation, expected",
        [
            ("add", [3, 7]),
            ("subtract", [-1, -1]),
            ("multiply", [2, 12]),
            ("divide", [0.5, 0.75]),
            ("floor_divide", [0, 0]),
            ("modulo", [1, 3]),
            ("power", [1, 81]),
        ],
    )
    def test_binary_operation_applied_elementwise(self, operation, expected):
        df = pd.DataFrame({"a": [1, 3], "b": [2, 4]})

        result = apply_arithmetic_operation_over_columns(
            df, columns=["a", "b"], output_column="out", operation=operation
        )

        assert result["out"].tolist() == pytest.approx(expected)

    def test_binary_operation_folds_left_to_right_across_three_columns(self):
        # Subtraction is non-associative, so this only passes if the columns
        # are folded strictly left-to-right: (10 - 2) - 3 = 5.
        df = pd.DataFrame({"a": [10], "b": [2], "c": [3]})

        result = apply_arithmetic_operation_over_columns(
            df, columns=["a", "b", "c"], output_column="out", operation="subtract"
        )

        assert result["out"].tolist() == [5]

    @pytest.mark.parametrize(
        "operation, expected",
        [
            ("min", [1, 2]),
            ("max", [5, 4]),
            ("mean", [3.0, 3.0]),
        ],
    )
    def test_aggregate_operation_reduces_row_wise_across_all_columns(self, operation, expected):
        df = pd.DataFrame({"a": [1, 4], "b": [5, 2], "c": [3, 3]})

        result = apply_arithmetic_operation_over_columns(
            df, columns=["a", "b", "c"], output_column="out", operation=operation
        )

        assert result["out"].tolist() == pytest.approx(expected)

    def test_output_column_added_without_disturbing_input_columns(self):
        df = pd.DataFrame({"a": [1], "b": [2]})

        result = apply_arithmetic_operation_over_columns(df, columns=["a", "b"], output_column="sum", operation="add")

        assert result["a"].tolist() == [1]
        assert result["b"].tolist() == [2]
        assert result["sum"].tolist() == [3]

    def test_unknown_operation_raises_key_error(self):
        df = pd.DataFrame({"a": [1], "b": [2]})

        with pytest.raises(KeyError):
            apply_arithmetic_operation_over_columns(
                df, columns=["a", "b"], output_column="out", operation="unsupported"
            )

    def test_missing_column_raises_key_error(self):
        df = pd.DataFrame({"a": [1], "b": [2]})

        with pytest.raises(KeyError):
            apply_arithmetic_operation_over_columns(
                df, columns=["a", "does_not_exist"], output_column="out", operation="add"
            )
