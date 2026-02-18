import pytest
import numpy as np
import pandas as pd

from ecoscope_workflows_ext_ate.tasks._stats import perform_anova_analysis 


# ──────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────

@pytest.fixture
def basic_df():
    """Well-formed dataframe with two categorical factors and a numeric target."""
    np.random.seed(42)
    n = 120
    return pd.DataFrame({
        "score":   np.random.normal(loc=50, scale=10, size=n),
        "group":   np.tile(["A", "B", "C"], n // 3),
        "region":  np.tile(["North", "South"], n // 2),
    })


@pytest.fixture
def df_with_nans(basic_df):
    """Same dataframe but with some NaN values scattered in."""
    df = basic_df.copy()
    df.loc[[0, 5, 10], "score"] = np.nan
    df.loc[[2, 8],     "group"] = np.nan
    return df


@pytest.fixture
def single_factor_df():
    """Minimal dataframe with one factor column."""
    np.random.seed(0)
    return pd.DataFrame({
        "value": np.random.normal(size=60),
        "category": np.tile(["X", "Y", "Z"], 20),
    })


# ──────────────────────────────────────────────
# Happy-path tests
# ──────────────────────────────────────────────

class TestReturnsValidDataFrame:
    def test_returns_dataframe(self, basic_df):
        result = perform_anova_analysis(basic_df, "score", ["group", "region"])
        assert isinstance(result, pd.DataFrame)

    def test_has_factor_column(self, basic_df):
        result = perform_anova_analysis(basic_df, "score", ["group", "region"])
        assert "factor" in result.columns

    def test_factor_names_cleaned(self, basic_df):
        """C(group) wrapper should be stripped from factor names."""
        result = perform_anova_analysis(basic_df, "score", ["group", "region"])
        factors = result["factor"].tolist()
        assert all("C(" not in f for f in factors), f"Unexpected C() wrapper in: {factors}"

    def test_expected_factors_present(self, basic_df):
        result = perform_anova_analysis(basic_df, "score", ["group", "region"])
        factors = result["factor"].tolist()
        assert "group"  in factors
        assert "region" in factors

    def test_single_factor(self, single_factor_df):
        result = perform_anova_analysis(single_factor_df, "value", ["category"])
        assert "category" in result["factor"].tolist()

    def test_p_values_between_0_and_1(self, basic_df):
        result = perform_anova_analysis(basic_df, "score", ["group", "region"])
        pval_col = next(c for c in result.columns if "PR(" in c or "p" in c.lower())
        valid = result[pval_col].dropna()
        assert (valid >= 0).all() and (valid <= 1).all()


class TestAnovaTypes:
    @pytest.mark.parametrize("anova_type", [1, 2, 3])
    def test_all_anova_types_succeed(self, basic_df, anova_type):
        result = perform_anova_analysis(
            basic_df, "score", ["group", "region"], anova_type=anova_type
        )
        assert isinstance(result, pd.DataFrame)
        assert not result.empty

    def test_default_type_is_2(self, basic_df):
        """Calling without anova_type should behave the same as anova_type=2."""
        result_default = perform_anova_analysis(basic_df, "score", ["group"])
        result_type2   = perform_anova_analysis(basic_df, "score", ["group"], anova_type=2)
        pd.testing.assert_frame_equal(result_default, result_type2)


class TestNaNHandling:
    def test_runs_with_nan_values(self, df_with_nans):
        """NaN rows should be dropped silently without raising."""
        result = perform_anova_analysis(df_with_nans, "score", ["group", "region"])
        assert isinstance(result, pd.DataFrame)
        assert not result.empty

    def test_nan_rows_are_excluded(self, basic_df, df_with_nans):
        """Results may differ when NaNs are present (fewer rows used)."""
        result_clean = perform_anova_analysis(basic_df,     "score", ["group"])
        result_nan   = perform_anova_analysis(df_with_nans, "score", ["group"])
        # Both should succeed but need not be identical
        assert isinstance(result_clean, pd.DataFrame)
        assert isinstance(result_nan,   pd.DataFrame)


# ──────────────────────────────────────────────
# Error / edge-case tests
# ──────────────────────────────────────────────

class TestValidationErrors:
    def test_missing_target_column(self, basic_df):
        with pytest.raises(ValueError, match="Target column 'nonexistent' not found"):
            perform_anova_analysis(basic_df, "nonexistent", ["group"])

    def test_missing_factor_column(self, basic_df):
        with pytest.raises(ValueError, match="Factor columns not found"):
            perform_anova_analysis(basic_df, "score", ["nonexistent_factor"])

    def test_partially_missing_factor_columns(self, basic_df):
        with pytest.raises(ValueError, match="Factor columns not found"):
            perform_anova_analysis(basic_df, "score", ["group", "bad_col"])

    def test_all_nan_target_raises(self, basic_df):
        df = basic_df.copy()
        df["score"] = np.nan
        with pytest.raises(ValueError, match="No valid data"):
            perform_anova_analysis(df, "score", ["group"])

    def test_empty_dataframe_raises(self):
        df = pd.DataFrame({"score": [], "group": []})
        with pytest.raises(ValueError):
            perform_anova_analysis(df, "score", ["group"])

    def test_single_level_factor_raises(self, basic_df):
        """A factor with only one unique level can't be used in ANOVA."""
        df = basic_df.copy()
        df["constant_group"] = "A"
        with pytest.raises(ValueError):
            perform_anova_analysis(df, "score", ["constant_group"])


# ──────────────────────────────────────────────
# Output structure tests
# ──────────────────────────────────────────────

class TestOutputStructure:
    def test_reset_index_applied(self, basic_df):
        """Result should have a default integer index (not the factor as index)."""
        result = perform_anova_analysis(basic_df, "score", ["group"])
        assert result.index.tolist() == list(range(len(result)))

    def test_no_c_wrapper_in_residual(self, basic_df):
        result = perform_anova_analysis(basic_df, "score", ["group", "region"])
        for f in result["factor"]:
            assert not f.startswith("C(")

    def test_numeric_stat_columns_are_float(self, basic_df):
        result = perform_anova_analysis(basic_df, "score", ["group"])
        numeric_cols = result.select_dtypes(include=[np.number]).columns
        assert len(numeric_cols) > 0, "Expected at least one numeric statistics column"