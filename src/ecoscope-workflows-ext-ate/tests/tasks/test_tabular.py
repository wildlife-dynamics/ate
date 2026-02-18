import pytest
import numpy as np
import pandas as pd

from ecoscope_workflows_ext_ate.tasks._tabular import (  # update import path as needed
    convert_object_to_value,
    convert_object_to_string,
    format_demographic_table,
    map_survey_responses,
    fill_missing_values,
    calculate_elephant_sentiment_score,
    exclude_value,
)


# ══════════════════════════════════════════════════════════════════
# Shared Fixtures
# ══════════════════════════════════════════════════════════════════


@pytest.fixture
def mixed_df():
    return pd.DataFrame(
        {
            "age": ["25", "30", "forty", None, "22"],
            "income": ["1000", "2000.5", "abc", "3000", None],
            "category": ["A", "B", "C", "A", "B"],
        }
    )


@pytest.fixture
def sentiment_df():
    return pd.DataFrame(
        {
            "q1": ["Strongly agree", "Agree", "Neutral", "Disagree", "Strongly disagree"],
            "q2": ["Agree", "Neutral", "Strongly agree", "Strongly disagree", "I dont know"],
            "q3": ["Strongly disagree", "Disagree", "Neutral", "Agree", "Strongly agree"],
        }
    )


@pytest.fixture
def demographic_df():
    np.random.seed(1)
    return pd.DataFrame(
        {
            "age": np.random.randint(18, 70, size=100),
            "gender": np.random.choice(["Male", "Female", "Other"], size=100),
            "region": np.random.choice(["North", "South", "East", "West"], size=100),
        }
    )


# ══════════════════════════════════════════════════════════════════
# convert_object_to_value
# ══════════════════════════════════════════════════════════════════


class TestConvertObjectToValue:
    def test_converts_numeric_strings(self, mixed_df):
        result = convert_object_to_value(mixed_df.copy(), columns=["age"])
        assert pd.api.types.is_float_dtype(result["age"]) or pd.api.types.is_numeric_dtype(result["age"])

    def test_non_convertible_become_nan(self, mixed_df):
        result = convert_object_to_value(mixed_df.copy(), columns=["age"])
        assert pd.isna(result.loc[2, "age"])  # "forty" → NaN

    def test_none_values_become_nan(self, mixed_df):
        result = convert_object_to_value(mixed_df.copy(), columns=["age"])
        assert pd.isna(result.loc[3, "age"])

    def test_valid_values_preserved(self, mixed_df):
        result = convert_object_to_value(mixed_df.copy(), columns=["age"])
        assert result.loc[0, "age"] == 25.0

    def test_default_converts_all_object_cols(self, mixed_df):
        result = convert_object_to_value(mixed_df.copy())
        for col in ["age", "income"]:
            assert pd.api.types.is_numeric_dtype(result[col])

    def test_multiple_columns(self, mixed_df):
        result = convert_object_to_value(mixed_df.copy(), columns=["age", "income"])
        assert pd.api.types.is_numeric_dtype(result["age"])
        assert pd.api.types.is_numeric_dtype(result["income"])

    def test_missing_column_raises(self, mixed_df):
        with pytest.raises(ValueError, match="Column not found: nonexistent"):
            convert_object_to_value(mixed_df.copy(), columns=["nonexistent"])

    def test_none_dataframe_raises(self):
        with pytest.raises(ValueError, match="df must be a pandas DataFrame"):
            convert_object_to_value(None)

    def test_non_dataframe_raises(self):
        with pytest.raises(ValueError):
            convert_object_to_value([1, 2, 3])

    def test_single_column_as_string(self, mixed_df):
        result = convert_object_to_value(mixed_df.copy(), columns="age")
        assert pd.api.types.is_numeric_dtype(result["age"])

    def test_non_object_columns_unaffected(self, mixed_df):
        df = mixed_df.copy()
        df["numeric_col"] = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = convert_object_to_value(df, columns=["age"])
        assert result["numeric_col"].tolist() == [1.0, 2.0, 3.0, 4.0, 5.0]


# ══════════════════════════════════════════════════════════════════
# convert_object_to_string
# ══════════════════════════════════════════════════════════════════


class TestConvertObjectToString:
    def test_converts_to_string_dtype(self, mixed_df):
        result = convert_object_to_string(mixed_df.copy(), columns=["age"])
        assert pd.api.types.is_string_dtype(result["age"])

    def test_default_converts_all_object_cols(self, mixed_df):
        result = convert_object_to_string(mixed_df.copy())
        for col in ["age", "income", "category"]:
            assert pd.api.types.is_string_dtype(result[col])

    def test_missing_column_raises(self, mixed_df):
        with pytest.raises(ValueError, match="Column not found: bad_col"):
            convert_object_to_string(mixed_df.copy(), columns=["bad_col"])

    def test_none_dataframe_raises(self):
        with pytest.raises(ValueError, match="df must be a pandas DataFrame"):
            convert_object_to_string(None)

    def test_single_column_as_string(self, mixed_df):
        result = convert_object_to_string(mixed_df.copy(), columns="category")
        assert pd.api.types.is_string_dtype(result["category"])

    def test_values_preserved_as_strings(self):
        df = pd.DataFrame({"x": ["hello", "world"]})
        result = convert_object_to_string(df, columns=["x"])
        assert result.loc[0, "x"] == "hello"

    def test_multiple_columns(self, mixed_df):
        result = convert_object_to_string(mixed_df.copy(), columns=["age", "category"])
        assert pd.api.types.is_string_dtype(result["age"])
        assert pd.api.types.is_string_dtype(result["category"])


# ══════════════════════════════════════════════════════════════════
# format_demographic_table
# ══════════════════════════════════════════════════════════════════


class TestFormatDemographicTable:
    def test_returns_dataframe(self, demographic_df):
        result = format_demographic_table(demographic_df, ["age", "gender"])
        assert isinstance(result, pd.DataFrame)

    def test_expected_columns_present(self, demographic_df):
        result = format_demographic_table(demographic_df, ["gender"])
        assert set(result.columns) == {"Demographic Variable", "Categories", "Number of responses"}

    def test_categorical_column_value_counts(self, demographic_df):
        result = format_demographic_table(demographic_df, ["gender"])
        assert "gender" in result["Demographic Variable"].values

    def test_numeric_column_bins_and_stats(self, demographic_df):
        result = format_demographic_table(demographic_df, ["age"])
        responses = result["Number of responses"].tolist()
        # Should contain a stats row with mean/median
        stats_rows = [r for r in responses if "mean=" in str(r)]
        assert len(stats_rows) == 1

    def test_missing_column_skipped(self, demographic_df):
        result = format_demographic_table(demographic_df, ["gender", "nonexistent"])
        assert "nonexistent" not in result["Demographic Variable"].values

    def test_nan_values_filled_as_no_response(self):
        df = pd.DataFrame({"color": ["Red", None, "Blue", None, "Red"]})
        result = format_demographic_table(df, ["color"])
        categories = result["Categories"].tolist()
        assert "No Response" in categories

    def test_demographic_variable_deduplicated(self, demographic_df):
        """Repeated variable name should be blanked out after first row."""
        result = format_demographic_table(demographic_df, ["gender"])
        var_col = result["Demographic Variable"].tolist()
        gender_rows = [v for v in var_col if v == "gender"]
        assert len(gender_rows) == 1  # Only the first row should have the name

    def test_percentage_format_in_responses(self, demographic_df):
        result = format_demographic_table(demographic_df, ["gender"])
        responses = result[result["Demographic Variable"] == "gender"]["Number of responses"]
        assert all("%" in str(r) for r in responses)

    def test_empty_columns_list_returns_empty_df(self, demographic_df):
        result = format_demographic_table(demographic_df, [])
        assert result.empty

    def test_all_columns_missing_returns_empty_df(self, demographic_df):
        result = format_demographic_table(demographic_df, ["col_x", "col_y"])
        assert result.empty

    def test_constant_numeric_column(self):
        df = pd.DataFrame({"score": [5, 5, 5, 5, 5], "label": ["a"] * 5})
        result = format_demographic_table(df, ["score"])
        assert isinstance(result, pd.DataFrame)


# ══════════════════════════════════════════════════════════════════
# map_survey_responses
# ══════════════════════════════════════════════════════════════════


class TestMapSurveyResponses:
    @pytest.fixture
    def survey_df(self):
        return pd.DataFrame(
            {
                "q1": ["yes", "no", "i_dont_know", "yes"],
                "q2": ["no", "yes", "prefer_not_to_answer", None],
            }
        )

    def test_maps_values_correctly(self, survey_df):
        value_map = {"yes": "Yes", "no": "No"}
        result = map_survey_responses(survey_df.copy(), ["q1"], value_map)
        assert result.loc[0, "q1"] == "Yes"
        assert result.loc[1, "q1"] == "No"

    def test_unmapped_values_preserved(self, survey_df):
        value_map = {"yes": "Yes"}
        result = map_survey_responses(survey_df.copy(), ["q1"], value_map)
        assert result.loc[1, "q1"] == "no"  # Not in map, preserved

    def test_inplace_false_does_not_modify_original(self, survey_df):
        original = survey_df.copy()
        map_survey_responses(survey_df, ["q1"], {"yes": "Yes"}, inplace=False)
        pd.testing.assert_frame_equal(survey_df, original)

    def test_inplace_true_modifies_original(self, survey_df):
        map_survey_responses(survey_df, ["q1"], {"yes": "Yes"}, inplace=True)
        assert survey_df.loc[0, "q1"] == "Yes"

    def test_missing_column_skipped_with_warning(self, survey_df, capsys):
        map_survey_responses(survey_df.copy(), ["nonexistent"], {"yes": "Yes"})
        captured = capsys.readouterr()
        assert "nonexistent" in captured.out

    def test_multiple_columns_mapped(self, survey_df):
        value_map = {"yes": "Yes", "no": "No"}
        result = map_survey_responses(survey_df.copy(), ["q1", "q2"], value_map)
        assert result.loc[0, "q1"] == "Yes"
        assert result.loc[0, "q2"] == "No"

    def test_none_values_preserved(self, survey_df):
        value_map = {"yes": "Yes"}
        result = map_survey_responses(survey_df.copy(), ["q2"], value_map)
        assert pd.isna(result.loc[3, "q2"])

    def test_returns_dataframe(self, survey_df):
        result = map_survey_responses(survey_df.copy(), ["q1"], {"yes": "Yes"})
        assert isinstance(result, pd.DataFrame)


# ══════════════════════════════════════════════════════════════════
# fill_missing_values
# ══════════════════════════════════════════════════════════════════


class TestFillMissingValues:
    @pytest.fixture
    def nulls_df(self):
        return pd.DataFrame(
            {
                "score": [1.0, None, 3.0, None, 5.0],
                "count": [10, None, 30, None, 50],
                "category": ["A", None, "B", None, "C"],
                "label": ["x", "y", None, "z", None],
            }
        )

    def test_fills_numeric_with_default_zero(self, nulls_df):
        result = fill_missing_values(nulls_df)
        assert result["score"].isna().sum() == 0
        assert result.loc[1, "score"] == 0

    def test_fills_categorical_with_default_unspecified(self, nulls_df):
        result = fill_missing_values(nulls_df)
        assert result["category"].isna().sum() == 0
        assert result.loc[1, "category"] == "Unspecified"

    def test_custom_numeric_fill_value(self, nulls_df):
        result = fill_missing_values(nulls_df, numeric_fill_value=-1)
        assert result.loc[1, "score"] == -1

    def test_custom_categorical_fill_value(self, nulls_df):
        result = fill_missing_values(nulls_df, categorical_fill_value="Unknown")
        assert result.loc[1, "category"] == "Unknown"

    def test_specific_numeric_columns(self, nulls_df):
        result = fill_missing_values(nulls_df, numeric_columns=["score"])
        assert result["score"].isna().sum() == 0
        assert result["count"].isna().sum() > 0  # untouched

    def test_specific_categorical_columns(self, nulls_df):
        result = fill_missing_values(nulls_df, categorical_columns=["category"])
        assert result["category"].isna().sum() == 0
        assert result["label"].isna().sum() > 0  # untouched

    def test_missing_numeric_column_raises(self, nulls_df):
        with pytest.raises(ValueError, match="Numeric columns not found"):
            fill_missing_values(nulls_df, numeric_columns=["bad_col"])

    def test_missing_categorical_column_raises(self, nulls_df):
        with pytest.raises(ValueError, match="Categorical columns not found"):
            fill_missing_values(nulls_df, categorical_columns=["bad_col"])

    def test_does_not_modify_original(self, nulls_df):
        original_nulls = nulls_df.isna().sum().sum()
        fill_missing_values(nulls_df)
        assert nulls_df.isna().sum().sum() == original_nulls

    def test_no_nulls_df_unchanged(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        result = fill_missing_values(df)
        pd.testing.assert_frame_equal(result, df)


# ══════════════════════════════════════════════════════════════════
# calculate_elephant_sentiment_score
# ══════════════════════════════════════════════════════════════════


class TestCalculateElephantSentimentScore:
    def test_returns_dataframe(self, sentiment_df):
        result = calculate_elephant_sentiment_score(sentiment_df, ["q1", "q2"], ["q3"])
        assert isinstance(result, pd.DataFrame)

    def test_score_columns_created(self, sentiment_df):
        result = calculate_elephant_sentiment_score(sentiment_df, ["q1"], ["q3"])
        assert "q1_score" in result.columns
        assert "q3_score" in result.columns

    def test_sentiment_score_column_exists(self, sentiment_df):
        result = calculate_elephant_sentiment_score(sentiment_df, ["q1", "q2"], ["q3"])
        assert "elephant_sentiment_score" in result.columns

    def test_overall_attitude_column_exists(self, sentiment_df):
        result = calculate_elephant_sentiment_score(sentiment_df, ["q1"], ["q3"])
        assert "overall_attitude" in result.columns

    def test_valid_response_count_column_exists(self, sentiment_df):
        result = calculate_elephant_sentiment_score(sentiment_df, ["q1"], ["q3"])
        assert "valid_response_count" in result.columns

    def test_positive_mapping_strongly_agree_is_5(self):
        df = pd.DataFrame({"q1": ["Strongly agree"]})
        result = calculate_elephant_sentiment_score(df, ["q1"], [])
        assert result.loc[result.index[0], "q1_score"] == 5

    def test_negative_mapping_strongly_agree_is_1(self):
        df = pd.DataFrame({"q1": ["Strongly agree"]})
        result = calculate_elephant_sentiment_score(df, [], ["q1"])
        assert result.loc[result.index[0], "q1_score"] == 1

    def test_positive_mapping_strongly_disagree_is_1(self):
        df = pd.DataFrame({"q1": ["Strongly disagree"]})
        result = calculate_elephant_sentiment_score(df, ["q1"], [])
        assert result.loc[result.index[0], "q1_score"] == 1

    def test_negative_mapping_strongly_disagree_is_5(self):
        df = pd.DataFrame({"q1": ["Strongly disagree"]})
        result = calculate_elephant_sentiment_score(df, [], ["q1"])
        assert result.loc[result.index[0], "q1_score"] == 5

    def test_i_dont_know_maps_to_zero(self):
        df = pd.DataFrame({"q1": ["I dont know"]})
        result = calculate_elephant_sentiment_score(df, ["q1"], [])
        assert result.loc[result.index[0], "q1_score"] == 0

    def test_score_is_mean_of_individual_scores(self):
        df = pd.DataFrame({"q1": ["Strongly agree"], "q2": ["Strongly disagree"]})
        result = calculate_elephant_sentiment_score(df, ["q1", "q2"], [])
        # q1=5, q2=1 → mean=3.0
        assert result["elephant_sentiment_score"].iloc[0] == 3.0

    def test_score_is_rounded_to_2_decimals(self, sentiment_df):
        result = calculate_elephant_sentiment_score(sentiment_df, ["q1", "q2"], ["q3"])
        scores = result["elephant_sentiment_score"].dropna()
        assert all(round(s, 2) == s for s in scores)

    def test_overall_attitude_strongly_agree_threshold(self):
        df = pd.DataFrame({"q1": ["Strongly agree"]})
        result = calculate_elephant_sentiment_score(df, ["q1"], [])
        assert result["overall_attitude"].iloc[0] == "Strongly agree"

    def test_overall_attitude_neutral_threshold(self):
        df = pd.DataFrame({"q1": ["Neutral"]})
        result = calculate_elephant_sentiment_score(df, ["q1"], [])
        assert result["overall_attitude"].iloc[0] == "Neutral"

    def test_result_sorted_by_score(self, sentiment_df):
        result = calculate_elephant_sentiment_score(sentiment_df, ["q1", "q2"], ["q3"])
        scores = result["elephant_sentiment_score"].dropna().tolist()
        assert scores == sorted(scores)

    def test_missing_column_warns(self, sentiment_df, capsys):
        calculate_elephant_sentiment_score(sentiment_df, ["nonexistent_col"], [])
        captured = capsys.readouterr()
        assert "nonexistent_col" in captured.out

    def test_does_not_modify_original(self, sentiment_df):
        original = sentiment_df.copy()
        calculate_elephant_sentiment_score(sentiment_df, ["q1"], ["q3"])
        pd.testing.assert_frame_equal(sentiment_df, original)

    def test_score_integer_dtype(self, sentiment_df):
        result = calculate_elephant_sentiment_score(sentiment_df, ["q1"], ["q3"])
        assert result["q1_score"].dtype == pd.Int64Dtype()

    @pytest.mark.parametrize(
        "attitude,expected",
        [
            ("Strongly agree", "Strongly agree"),
            ("Agree", "Agree"),
            ("Neutral", "Neutral"),
            ("Disagree", "Disagree"),
            ("Strongly disagree", "Strongly disagree"),
        ],
    )
    def test_overall_attitude_all_labels(self, attitude, expected):
        df = pd.DataFrame({"q1": [attitude]})
        result = calculate_elephant_sentiment_score(df, ["q1"], [])
        assert result["overall_attitude"].iloc[0] == expected


# ══════════════════════════════════════════════════════════════════
# exclude_value
# ══════════════════════════════════════════════════════════════════


class TestExcludeValue:
    @pytest.fixture
    def base_df(self):
        return pd.DataFrame(
            {
                "status": ["active", "inactive", "active", "pending", "inactive"],
                "count": [10, 20, 30, 40, 50],
            }
        )

    def test_excludes_string_value(self, base_df):
        result = exclude_value(base_df, "status", "inactive")
        assert "inactive" not in result["status"].values

    def test_excludes_numeric_value(self, base_df):
        result = exclude_value(base_df, "count", 20)
        assert 20 not in result["count"].values

    def test_correct_rows_remain(self, base_df):
        result = exclude_value(base_df, "status", "inactive")
        assert len(result) == 3

    def test_does_not_modify_original(self, base_df):
        original_len = len(base_df)
        exclude_value(base_df, "status", "inactive")
        assert len(base_df) == original_len

    def test_returns_dataframe(self, base_df):
        result = exclude_value(base_df, "status", "active")
        assert isinstance(result, pd.DataFrame)

    def test_no_matching_value_returns_all_rows(self, base_df):
        result = exclude_value(base_df, "status", "nonexistent")
        assert len(result) == len(base_df)

    def test_all_matching_returns_empty(self, base_df):
        result = exclude_value(base_df, "status", "active")
        assert len(result[result["status"] == "active"]) == 0

    def test_excludes_float_value(self):
        df = pd.DataFrame({"val": [1.5, 2.5, 3.5, 2.5]})
        result = exclude_value(df, "val", 2.5)
        assert 2.5 not in result["val"].values
        assert len(result) == 2

    def test_index_reset_not_needed_but_data_consistent(self, base_df):
        result = exclude_value(base_df, "status", "inactive")
        assert result["count"].tolist() == [10, 30, 40]
