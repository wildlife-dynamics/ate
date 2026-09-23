"""Tests for ecoscope_workflows_ext_ate.tasks.transformation._stats.

`compute_tukey_comparisons` is registered via `wt_registry.register()`, a
no-op decorator at call time, so it is called directly as plain Python
against small, hand-built DataFrames. It delegates the actual Tukey HSD
computation to the real `statsmodels.stats.multicomp.pairwise_tukeyhsd`
(installed and cheap to run for real), so results below are checked
against group means chosen to make significance unambiguous rather than
mocked.
"""

from __future__ import annotations

import pandas as pd
import pytest

from ecoscope_workflows_ext_ate.tasks.transformation._stats import compute_tukey_comparisons


@pytest.fixture
def three_group_df() -> pd.DataFrame:
    # Groups A and C sit close together; group B is far from both, so
    # A-B and B-C should come out significant while A-C should not.
    return pd.DataFrame(
        {
            "value": [1, 2, 3, 1.5, 2.5, 10, 11, 12, 10.5, 11.5, 1.1, 2.1, 3.1, 1.6, 2.6],
            "group": ["A"] * 5 + ["B"] * 5 + ["C"] * 5,
        }
    )


class TestComputeTukeyComparisons:
    def test_returns_expected_columns_sorted_by_mean_diff_ascending(self, three_group_df):
        result = compute_tukey_comparisons(dataframe=three_group_df, value_column="value", group_column="group")

        assert list(result.columns) == [
            "group1",
            "group2",
            "mean_diff",
            "ci_lower",
            "ci_upper",
            "p_value",
            "is_significant",
        ]
        assert result["mean_diff"].is_monotonic_increasing

    def test_one_row_per_pairwise_group_comparison(self, three_group_df):
        result = compute_tukey_comparisons(dataframe=three_group_df, value_column="value", group_column="group")

        # C(3, 2) = 3 pairwise comparisons for 3 groups.
        assert len(result) == 3

    def test_far_apart_groups_are_flagged_significant(self, three_group_df):
        result = compute_tukey_comparisons(dataframe=three_group_df, value_column="value", group_column="group")

        by_pair = {frozenset((r["group1"], r["group2"])): r["is_significant"] for _, r in result.iterrows()}
        assert by_pair[frozenset(("A", "B"))]
        assert by_pair[frozenset(("B", "C"))]

    def test_close_groups_are_not_flagged_significant(self, three_group_df):
        result = compute_tukey_comparisons(dataframe=three_group_df, value_column="value", group_column="group")

        by_pair = {frozenset((r["group1"], r["group2"])): r["is_significant"] for _, r in result.iterrows()}
        assert not by_pair[frozenset(("A", "C"))]

    def test_is_significant_column_is_boolean_dtype(self, three_group_df):
        result = compute_tukey_comparisons(dataframe=three_group_df, value_column="value", group_column="group")

        assert result["is_significant"].dtype == bool

    def test_same_value_and_group_column_raises_value_error(self, three_group_df):
        with pytest.raises(ValueError, match="must be different columns"):
            compute_tukey_comparisons(dataframe=three_group_df, value_column="group", group_column="group")

    def test_non_numeric_value_column_raises_value_error(self):
        df = pd.DataFrame({"value": ["a", "b", "c"], "group": ["A", "A", "B"]})

        with pytest.raises(ValueError, match="must be numeric"):
            compute_tukey_comparisons(dataframe=df, value_column="value", group_column="group")

    def test_all_null_data_raises_value_error(self):
        # Must stay a numeric dtype (float NaN, not None/object) so it gets
        # past the "must be numeric" check and hits the dropna-emptiness one.
        df = pd.DataFrame({"value": [float("nan"), float("nan")], "group": ["A", "B"]})

        with pytest.raises(ValueError, match="No valid data"):
            compute_tukey_comparisons(dataframe=df, value_column="value", group_column="group")

    def test_fewer_than_two_groups_raises_value_error(self):
        df = pd.DataFrame({"value": [1, 2, 3], "group": ["A", "A", "A"]})

        with pytest.raises(ValueError, match="Need at least 2 groups"):
            compute_tukey_comparisons(dataframe=df, value_column="value", group_column="group")

    def test_higher_confidence_level_widens_the_interval(self, three_group_df):
        high_confidence = compute_tukey_comparisons(
            dataframe=three_group_df, value_column="value", group_column="group", confidence_level=0.99
        )
        low_confidence = compute_tukey_comparisons(
            dataframe=three_group_df, value_column="value", group_column="group", confidence_level=0.80
        )

        high_width = (high_confidence["ci_upper"] - high_confidence["ci_lower"]).iloc[0]
        low_width = (low_confidence["ci_upper"] - low_confidence["ci_lower"]).iloc[0]
        assert high_width > low_width
