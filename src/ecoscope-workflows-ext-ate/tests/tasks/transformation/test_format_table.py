"""Tests for ecoscope_workflows_ext_ate.tasks.transformation._format_table.

`format_demographic_table` is registered via `wt_registry.register()`, a
no-op decorator at call time, so it is called directly as plain Python
against small, hand-built DataFrames.
"""

from __future__ import annotations

import pandas as pd

from ecoscope_workflows_ext_ate.tasks.transformation._format_table import format_demographic_table


class TestFormatDemographicTable:
    def test_categorical_column_reports_percentage_and_count_per_value(self):
        df = pd.DataFrame({"gender": ["M", "F", "M", "F"]})

        result = format_demographic_table(df, columns_of_interest=["gender"])

        rows = {r["Categories"]: r["Number of responses"] for _, r in result.iterrows()}
        assert rows["M"] == "50.00% (n=2)"
        assert rows["F"] == "50.00% (n=2)"

    def test_categorical_nulls_are_grouped_as_no_response(self):
        df = pd.DataFrame({"gender": ["M", "F", None, None]})

        result = format_demographic_table(df, columns_of_interest=["gender"])

        rows = {r["Categories"]: r["Number of responses"] for _, r in result.iterrows()}
        assert rows["No Response"] == "50.00% (n=2)"

    def test_numeric_column_is_binned_with_a_trailing_stats_row(self):
        df = pd.DataFrame({"age": [20, 25, 30, 35, 40, 45, 50, 55, 60, 65]})

        result = format_demographic_table(df, columns_of_interest=["age"])

        # 5 quantile bins + 1 trailing stats row.
        assert len(result) == 6
        stats_row = result.iloc[-1]
        assert stats_row["Categories"] == ""
        assert "mean=42.5" in stats_row["Number of responses"]
        assert "median=42" in stats_row["Number of responses"]
        assert "min=20" in stats_row["Number of responses"]
        assert "max=65" in stats_row["Number of responses"]

    def test_numeric_bins_cover_the_full_population(self):
        df = pd.DataFrame({"age": [20, 25, 30, 35, 40, 45, 50, 55, 60, 65]})

        result = format_demographic_table(df, columns_of_interest=["age"])

        bin_rows = result.iloc[:-1]
        total_n = sum(int(s.split("n=")[1].rstrip(")")) for s in bin_rows["Number of responses"])
        assert total_n == len(df)

    def test_numeric_column_with_a_single_distinct_value_gets_one_bin(self):
        df = pd.DataFrame({"age": [30, 30, 30]})

        result = format_demographic_table(df, columns_of_interest=["age"])

        assert result.iloc[0]["Categories"] == "30"
        assert result.iloc[0]["Number of responses"] == "100.00% (n=3)"

    def test_column_more_than_half_numeric_is_treated_as_numeric(self):
        df = pd.DataFrame({"score": ["1", "2", "3", "not_a_number", "5"]})

        result = format_demographic_table(df, columns_of_interest=["score"])

        # Binned as numeric (bin-range labels), not one categorical row per literal string.
        assert result.iloc[0]["Categories"] != "1"
        assert any("mean=" in v for v in result["Number of responses"])

    def test_column_half_or_less_numeric_is_treated_as_categorical(self):
        df = pd.DataFrame({"mixed": ["1", "a", "b", "c"]})

        result = format_demographic_table(df, columns_of_interest=["mixed"])

        assert set(result["Categories"]) == {"1", "a", "b", "c"}

    def test_columns_not_present_in_dataframe_are_skipped(self):
        df = pd.DataFrame({"x": [1, 2, 3]})

        result = format_demographic_table(df, columns_of_interest=["does_not_exist"])

        assert result.empty

    def test_only_the_first_row_of_each_variable_group_keeps_its_name(self):
        df = pd.DataFrame({"gender": ["M", "F"], "age": [20, 60]})

        result = format_demographic_table(df, columns_of_interest=["gender", "age"])

        var_names = result["Demographic Variable"].tolist()
        assert var_names.count("gender") == 1
        assert var_names.count("age") == 1

    def test_multiple_columns_processed_in_the_order_given(self):
        df = pd.DataFrame({"gender": ["M", "F"], "age": [20, 60]})

        result = format_demographic_table(df, columns_of_interest=["gender", "age"])

        non_blank_vars = [v for v in result["Demographic Variable"].tolist() if v]
        assert non_blank_vars == ["gender", "age"]

    def test_empty_columns_of_interest_returns_empty_dataframe(self):
        df = pd.DataFrame({"x": [1, 2, 3]})

        result = format_demographic_table(df, columns_of_interest=[])

        assert result.empty
