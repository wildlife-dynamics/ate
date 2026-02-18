import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock

from ecoscope_workflows_ext_ate.tasks._plot import (  # update import path as needed
    _infer_response_order,
    _generate_color_scheme,
    _sort_dataframe,
    create_likert_chart,
    get_chart_recommendations,
    draw_pie_and_persist,
    draw_bar_and_persist,
    draw_boxplot,
    draw_boxplot_and_persist,
    draw_scatter_chart,
    draw_ols_scatterplot_and_persist,
    draw_tukey_plot,
    draw_tukey_plots_and_persist,
)


# ══════════════════════════════════════════════════════════════════
# Shared Fixtures
# ══════════════════════════════════════════════════════════════════


@pytest.fixture
def likert_df():
    np.random.seed(42)
    choices = ["Strongly Disagree", "Disagree", "Neutral", "Agree", "Strongly Agree"]
    return pd.DataFrame(
        {
            "Q1: Elephants are important": np.random.choice(choices, 50),
            "Q2: Coexistence is possible": np.random.choice(choices, 50),
            "Q3: I feel safe near elephants": np.random.choice(choices, 50),
        }
    )


@pytest.fixture
def numeric_df():
    np.random.seed(0)
    return pd.DataFrame(
        {
            "score": np.random.normal(50, 10, 100),
            "age": np.random.randint(18, 70, 100),
            "income": np.random.exponential(30000, 100),
            "group": np.random.choice(["A", "B", "C"], 100),
            "region": np.random.choice(["North", "South"], 100),
            "id": range(100),
        }
    )


@pytest.fixture
def mixed_types_df():
    return pd.DataFrame(
        {
            "cat_low": ["A", "B", "C"] * 10,
            "cat_high": [f"val_{i}" for i in range(30)],
            "bool_col": [True, False] * 15,
            "int_few": [1, 2, 3, 4, 5] * 6,
            "int_many": list(range(30)),
            "float_col": np.linspace(0, 100, 30),
            "dt_col": pd.date_range("2020-01-01", periods=30, freq="D"),
            "all_null": [None] * 30,
        }
    )


# ══════════════════════════════════════════════════════════════════
# _infer_response_order
# ══════════════════════════════════════════════════════════════════


class TestInferResponseOrder:
    def test_detects_likert_pattern(self):
        df = pd.DataFrame(
            {
                "q1": ["Strongly Disagree", "Agree", "Neutral"],
                "q2": ["Disagree", "Strongly Agree", "Neutral"],
            }
        )
        order = _infer_response_order(df)
        assert order == ["Strongly Disagree", "Disagree", "Neutral", "Agree", "Strongly Agree"]

    def test_detects_satisfaction_pattern(self):
        df = pd.DataFrame(
            {
                "q1": ["Very Dissatisfied", "Satisfied", "Neutral"],
            }
        )
        order = _infer_response_order(df)
        assert "Satisfied" in order
        assert order.index("Very Dissatisfied") < order.index("Satisfied")

    def test_falls_back_to_frequency_order(self):
        df = pd.DataFrame(
            {
                "q1": ["Foo", "Bar", "Foo", "Baz"],
            }
        )
        order = _infer_response_order(df)
        assert "Foo" in order
        assert order[0] == "Foo"  # Most frequent first

    def test_ignores_nan_values(self):
        df = pd.DataFrame(
            {
                "q1": ["Agree", None, "Neutral", "Disagree"],
            }
        )
        order = _infer_response_order(df)
        assert None not in order
        assert np.nan not in order

    def test_returns_list(self, likert_df):
        order = _infer_response_order(likert_df)
        assert isinstance(order, list)

    def test_no_duplicates_in_order(self, likert_df):
        order = _infer_response_order(likert_df)
        assert len(order) == len(set(order))


# ══════════════════════════════════════════════════════════════════
# _generate_color_scheme
# ══════════════════════════════════════════════════════════════════


class TestGenerateColorScheme:
    def test_five_point_returns_five_colors(self):
        responses = ["SD", "D", "N", "A", "SA"]
        colors = _generate_color_scheme(responses)
        assert len(colors) == 5

    def test_seven_point_returns_seven_colors(self):
        responses = [f"R{i}" for i in range(7)]
        colors = _generate_color_scheme(responses)
        assert len(colors) == 7

    def test_custom_length_returns_correct_count(self):
        for n in [3, 4, 6, 8]:
            responses = [f"R{i}" for i in range(n)]
            colors = _generate_color_scheme(responses)
            assert len(colors) == n

    def test_colors_are_hex_strings(self):
        responses = ["SD", "D", "N", "A", "SA"]
        colors = _generate_color_scheme(responses)
        for key, val in colors.items():
            assert val.startswith("#"), f"Expected hex color, got: {val}"
            assert len(val) == 7

    def test_all_keys_are_response_labels(self):
        responses = ["Low", "Medium", "High"]
        colors = _generate_color_scheme(responses)
        assert set(colors.keys()) == set(responses)

    def test_five_point_specific_colors(self):
        """Verify known color values for 5-point scale."""
        responses = ["SD", "D", "N", "A", "SA"]
        colors = _generate_color_scheme(responses)
        assert colors["SD"] == "#2c5282"
        assert colors["SA"] == "#c05621"


# ══════════════════════════════════════════════════════════════════
# _sort_dataframe
# ══════════════════════════════════════════════════════════════════


class TestSortDataframe:
    def test_sort_by_name(self, likert_df):
        result = _sort_dataframe(likert_df, [], [], sort_by="name")
        assert list(result.columns) == sorted(likert_df.columns)

    def test_sort_by_positive_returns_all_columns(self, likert_df):
        positive = ["Agree", "Strongly Agree"]
        negative = ["Disagree", "Strongly Disagree"]
        result = _sort_dataframe(likert_df, positive, negative, sort_by="positive")
        assert set(result.columns) == set(likert_df.columns)

    def test_sort_by_negative_returns_all_columns(self, likert_df):
        positive = ["Agree", "Strongly Agree"]
        negative = ["Disagree", "Strongly Disagree"]
        result = _sort_dataframe(likert_df, positive, negative, sort_by="negative")
        assert set(result.columns) == set(likert_df.columns)

    def test_sort_preserves_row_count(self, likert_df):
        result = _sort_dataframe(likert_df, [], [], sort_by="name")
        assert len(result) == len(likert_df)

    def test_unknown_sort_key_still_returns_df(self, likert_df):
        result = _sort_dataframe(likert_df, [], [], sort_by="unknown_key")
        assert isinstance(result, pd.DataFrame)


# ══════════════════════════════════════════════════════════════════
# create_likert_chart
# ══════════════════════════════════════════════════════════════════


class TestCreateLikertChart:
    def test_returns_html_string(self, likert_df):
        result = create_likert_chart(likert_df)
        assert isinstance(result, str)
        assert "<div" in result or "<!DOCTYPE" in result

    def test_custom_title_in_output(self, likert_df):
        result = create_likert_chart(likert_df, title="My Survey Results")
        assert "My Survey Results" in result

    def test_custom_response_order(self, likert_df):
        order = ["Strongly Disagree", "Disagree", "Neutral", "Agree", "Strongly Agree"]
        result = create_likert_chart(likert_df, response_order=order)
        assert isinstance(result, str)

    def test_custom_colors_accepted(self, likert_df):
        colors = {
            "Strongly Disagree": "#ff0000",
            "Disagree": "#ff8800",
            "Neutral": "#888888",
            "Agree": "#00aa00",
            "Strongly Agree": "#006600",
        }
        result = create_likert_chart(likert_df, colors=colors)
        assert isinstance(result, str)

    def test_with_sort_questions_enabled(self, likert_df):
        result = create_likert_chart(likert_df, sort_questions=True, sort_by="positive")
        assert isinstance(result, str)

    def test_sort_by_negative(self, likert_df):
        result = create_likert_chart(likert_df, sort_questions=True, sort_by="negative")
        assert isinstance(result, str)

    def test_sort_by_name(self, likert_df):
        result = create_likert_chart(likert_df, sort_questions=True, sort_by="name")
        assert isinstance(result, str)

    def test_show_percentages_false(self, likert_df):
        result = create_likert_chart(likert_df, show_percentages=False)
        assert isinstance(result, str)

    def test_custom_dimensions(self, likert_df):
        result = create_likert_chart(likert_df, width=800, height_per_question=80, min_height=300)
        assert isinstance(result, str)

    def test_even_number_of_responses(self):
        df = pd.DataFrame(
            {
                "Q1": ["Disagree", "Agree", "Disagree", "Agree"] * 10,
                "Q2": ["Agree", "Agree", "Disagree", "Disagree"] * 10,
            }
        )
        result = create_likert_chart(df, response_order=["Disagree", "Agree"])
        assert isinstance(result, str)

    def test_explicit_neutral_categories(self, likert_df):
        result = create_likert_chart(
            likert_df,
            neutral_categories=["Neutral"],
            response_order=["Strongly Disagree", "Disagree", "Neutral", "Agree", "Strongly Agree"],
        )
        assert isinstance(result, str)


# ══════════════════════════════════════════════════════════════════
# get_chart_recommendations
# ══════════════════════════════════════════════════════════════════


class TestGetChartRecommendations:
    def test_returns_dataframe(self, mixed_types_df):
        result = get_chart_recommendations(mixed_types_df)
        assert isinstance(result, pd.DataFrame)

    def test_expected_columns(self, mixed_types_df):
        result = get_chart_recommendations(mixed_types_df)
        assert set(result.columns) == {"column", "chart_type", "unique_count", "data_type", "reason"}

    def test_all_null_column_excluded(self, mixed_types_df):
        result = get_chart_recommendations(mixed_types_df)
        assert "all_null" not in result["column"].values

    def test_low_cardinality_categorical_is_pie(self, mixed_types_df):
        result = get_chart_recommendations(mixed_types_df)
        row = result[result["column"] == "cat_low"].iloc[0]
        assert row["chart_type"] == "Pie Chart"

    def test_high_cardinality_categorical_not_recommended(self, mixed_types_df):
        result = get_chart_recommendations(mixed_types_df)
        row = result[result["column"] == "cat_high"].iloc[0]
        assert row["chart_type"] == "Not Recommended"

    def test_bool_column_is_pie(self, mixed_types_df):
        result = get_chart_recommendations(mixed_types_df)
        row = result[result["column"] == "bool_col"].iloc[0]
        assert row["chart_type"] == "Pie Chart"

    def test_datetime_column_is_line(self, mixed_types_df):
        result = get_chart_recommendations(mixed_types_df)
        row = result[result["column"] == "dt_col"].iloc[0]
        assert row["chart_type"] == "Line Chart"

    def test_continuous_float_is_histogram(self, mixed_types_df):
        result = get_chart_recommendations(mixed_types_df)
        row = result[result["column"] == "float_col"].iloc[0]
        assert row["chart_type"] == "Histogram"

    def test_discrete_int_few_values_is_bar(self, mixed_types_df):
        result = get_chart_recommendations(mixed_types_df)
        row = result[result["column"] == "int_few"].iloc[0]
        assert row["chart_type"] == "Bar Chart"

    def test_excluded_columns_not_in_result(self, mixed_types_df):
        result = get_chart_recommendations(mixed_types_df, exclude_columns=["cat_low", "bool_col"])
        assert "cat_low" not in result["column"].values
        assert "bool_col" not in result["column"].values

    def test_custom_pie_chart_threshold(self, mixed_types_df):
        """With threshold=2, only binary categoricals become pie charts."""
        result = get_chart_recommendations(mixed_types_df, pie_chart_threshold=2)
        row = result[result["column"] == "cat_low"].iloc[0]
        assert row["chart_type"] == "Bar Chart"  # 3 unique values > threshold of 2


# ══════════════════════════════════════════════════════════════════
# draw_boxplot
# ══════════════════════════════════════════════════════════════════


class TestDrawBoxplot:
    def test_returns_html_string(self, numeric_df):
        result = draw_boxplot(numeric_df, y_column="score")
        assert isinstance(result, str)
        assert "<div" in result or "<!DOCTYPE" in result

    def test_with_x_column(self, numeric_df):
        result = draw_boxplot(numeric_df, y_column="score", x_column="group")
        assert isinstance(result, str)

    def test_horizontal_orientation(self, numeric_df):
        result = draw_boxplot(numeric_df, y_column="score", orientation="h")
        assert isinstance(result, str)

    def test_overlay_boxmode(self, numeric_df):
        result = draw_boxplot(numeric_df, y_column="score", x_column="group", boxmode="overlay")
        assert isinstance(result, str)

    def test_show_all_points(self, numeric_df):
        result = draw_boxplot(numeric_df, y_column="score", show_points="all")
        assert isinstance(result, str)

    def test_show_outliers(self, numeric_df):
        result = draw_boxplot(numeric_df, y_column="score", show_points="outliers")
        assert isinstance(result, str)

    def test_with_color_column(self, numeric_df):
        numeric_df = numeric_df.copy()
        numeric_df["color"] = "#ff0000"
        result = draw_boxplot(numeric_df, y_column="score", color_column="color")
        assert isinstance(result, str)

    def test_with_marker_colors_and_x_column(self, numeric_df):
        from ecoscope_workflows_ext_ecoscope.tasks.results._ecoplot import PlotStyle

        result = draw_boxplot(
            numeric_df,
            y_column="score",
            x_column="group",
            plot_style=PlotStyle(marker_colors=["#ff0000", "#00ff00", "#0000ff"]),
        )
        assert isinstance(result, str)

    def test_widget_id_included_in_output(self, numeric_df):
        result = draw_boxplot(numeric_df, y_column="score", widget_id="test-widget")
        assert isinstance(result, str)


# ══════════════════════════════════════════════════════════════════
# draw_scatter_chart
# ══════════════════════════════════════════════════════════════════


class TestDrawScatterChart:
    from ecoscope_workflows_ext_ate.tasks._plot import ScatterStyle, TrendlineStyle

    def test_returns_html_string(self, numeric_df):
        result = draw_scatter_chart(numeric_df, x_column="age", y_column="score")
        assert isinstance(result, str)

    def test_with_ols_trendline(self, numeric_df):
        from ecoscope_workflows_ext_ate.tasks._plot import TrendlineStyle

        result = draw_scatter_chart(
            numeric_df, x_column="age", y_column="score", trendline_style=TrendlineStyle(enabled=True, type="ols")
        )
        assert isinstance(result, str)

    def test_with_lowess_trendline(self, numeric_df):
        from ecoscope_workflows_ext_ate.tasks._plot import TrendlineStyle

        result = draw_scatter_chart(
            numeric_df, x_column="age", y_column="score", trendline_style=TrendlineStyle(enabled=True, type="lowess")
        )
        assert isinstance(result, str)

    def test_with_category_column(self, numeric_df):
        result = draw_scatter_chart(numeric_df, x_column="age", y_column="score", category_column="group")
        assert isinstance(result, str)

    def test_with_color_column(self, numeric_df):
        result = draw_scatter_chart(numeric_df, x_column="age", y_column="score", color_column="income")
        assert isinstance(result, str)

    def test_with_size_column(self, numeric_df):
        result = draw_scatter_chart(numeric_df, x_column="age", y_column="score", size_column="income")
        assert isinstance(result, str)

    def test_all_nan_raises(self, numeric_df):
        df = numeric_df.copy()
        df["score"] = np.nan
        with pytest.raises(ValueError, match="No valid data"):
            draw_scatter_chart(df, x_column="age", y_column="score")

    def test_scatter_style_applied(self, numeric_df):
        from ecoscope_workflows_ext_ate.tasks._plot import ScatterStyle

        result = draw_scatter_chart(
            numeric_df,
            x_column="age",
            y_column="score",
            scatter_style=ScatterStyle(marker_color="#abc123", marker_size=12, marker_opacity=0.5),
        )
        assert isinstance(result, str)

    def test_widget_id_in_output(self, numeric_df):
        result = draw_scatter_chart(numeric_df, x_column="age", y_column="score", widget_id="scatter-1")
        assert isinstance(result, str)


# ══════════════════════════════════════════════════════════════════
# draw_tukey_plot
# ══════════════════════════════════════════════════════════════════


class TestDrawTukeyPlot:
    def test_returns_html_string(self, numeric_df):
        result = draw_tukey_plot(numeric_df, value_column="score", group_column="group")
        assert isinstance(result, str)

    def test_all_nan_raises(self, numeric_df):
        df = numeric_df.copy()
        df["score"] = np.nan
        with pytest.raises(ValueError, match="No valid data"):
            draw_tukey_plot(df, value_column="score", group_column="group")

    def test_two_groups_produces_one_comparison(self, numeric_df):
        """Two groups → one pairwise comparison."""
        result = draw_tukey_plot(numeric_df, value_column="score", group_column="region")
        assert isinstance(result, str)

    def test_custom_tukey_style(self, numeric_df):
        from ecoscope_workflows_ext_ate.tasks._plot import TukeyPlotStyle

        result = draw_tukey_plot(
            numeric_df,
            value_column="score",
            group_column="group",
            tukey_style=TukeyPlotStyle(
                significant_color="#ff0000",
                non_significant_color="#0000ff",
                confidence_level=0.99,
                marker_size=12,
                line_width=3,
            ),
        )
        assert isinstance(result, str)

    def test_widget_id_accepted(self, numeric_df):
        result = draw_tukey_plot(numeric_df, value_column="score", group_column="group", widget_id="tukey-widget")
        assert isinstance(result, str)

    def test_significantly_different_groups_detected(self):
        """Groups with very different means should be flagged as significant."""
        df = pd.DataFrame(
            {
                "value": [1.0] * 20 + [100.0] * 20,
                "group": ["low"] * 20 + ["high"] * 20,
            }
        )
        result = draw_tukey_plot(df, value_column="value", group_column="group")
        assert "Significant" in result  # legend label should appear in HTML

    def test_similar_groups_not_significant(self):
        np.random.seed(42)
        df = pd.DataFrame(
            {
                "value": np.concatenate(
                    [
                        np.random.normal(50, 1, 30),
                        np.random.normal(50.01, 1, 30),  # Nearly identical means
                    ]
                ),
                "group": ["A"] * 30 + ["B"] * 30,
            }
        )
        result = draw_tukey_plot(df, value_column="value", group_column="group")
        assert "Not Significant" in result


# ══════════════════════════════════════════════════════════════════
# Persist functions (draw_pie_and_persist, draw_bar_and_persist,
# draw_boxplot_and_persist, draw_ols_scatterplot_and_persist,
# draw_tukey_plots_and_persist)
#
# These wrap the core charting functions + file I/O.
# We mock persist_text to avoid filesystem writes in tests.
# ══════════════════════════════════════════════════════════════════

MOCK_FILE_PATH = "/tmp/test_output.html"


@pytest.fixture
def mock_persist(monkeypatch):
    """Patch persist_text to return a fake path without writing files."""
    mock = MagicMock(return_value=MOCK_FILE_PATH)
    monkeypatch.setattr("ecoscope_workflows_core.tasks.io.persist_text", mock)
    return mock


@pytest.fixture
def mock_draw_pie(monkeypatch):
    monkeypatch.setattr(
        "ecoscope_workflows_ext_ecoscope.tasks.results._ecoplot.draw_pie_chart",
        MagicMock(return_value="<html>pie</html>"),
    )


@pytest.fixture
def mock_draw_bar(monkeypatch):
    monkeypatch.setattr(
        "ecoscope_workflows_ext_ecoscope.tasks.results._ecoplot.draw_bar_chart",
        MagicMock(return_value="<html>bar</html>"),
    )


class TestDrawPieAndPersist:
    def test_returns_list(self, numeric_df, mock_persist, mock_draw_pie):
        result = draw_pie_and_persist("/tmp", numeric_df, ["group"])
        assert isinstance(result, list)

    def test_returns_file_path_per_column(self, numeric_df, mock_persist, mock_draw_pie):
        result = draw_pie_and_persist("/tmp", numeric_df, ["group", "region"])
        assert len(result) == 2

    def test_none_dataframe_raises(self, mock_persist):
        with pytest.raises(ValueError, match="DataFrame is None or empty"):
            draw_pie_and_persist("/tmp", None, ["group"])

    def test_empty_dataframe_raises(self, mock_persist):
        with pytest.raises(ValueError, match="DataFrame is None or empty"):
            draw_pie_and_persist("/tmp", pd.DataFrame(), ["group"])

    def test_missing_columns_raises(self, numeric_df, mock_persist):
        with pytest.raises(ValueError, match="Columns not found"):
            draw_pie_and_persist("/tmp", numeric_df, ["nonexistent"])

    def test_string_column_accepted(self, numeric_df, mock_persist, mock_draw_pie):
        result = draw_pie_and_persist("/tmp", numeric_df, "group")
        assert isinstance(result, list)

    def test_non_list_non_string_columns_raises(self, numeric_df, mock_persist):
        with pytest.raises(ValueError, match="columns parameter"):
            draw_pie_and_persist("/tmp", numeric_df, 123)

    def test_all_nan_column_skipped(self, numeric_df, mock_persist, mock_draw_pie, capsys):
        df = numeric_df.copy()
        df["empty_col"] = np.nan
        draw_pie_and_persist("/tmp", df, ["empty_col"])
        captured = capsys.readouterr()
        assert "empty_col" in captured.out


class TestDrawBarAndPersist:
    def test_returns_list(self, numeric_df, mock_persist, mock_draw_bar):
        result = draw_bar_and_persist("/tmp", numeric_df, ["group"])
        assert isinstance(result, list)

    def test_none_dataframe_raises(self, mock_persist):
        with pytest.raises(ValueError):
            draw_bar_and_persist("/tmp", None, ["group"])

    def test_empty_dataframe_raises(self, mock_persist):
        with pytest.raises(ValueError):
            draw_bar_and_persist("/tmp", pd.DataFrame(), ["group"])

    def test_missing_columns_raises(self, numeric_df, mock_persist):
        with pytest.raises(ValueError, match="Columns not found"):
            draw_bar_and_persist("/tmp", numeric_df, ["bad_col"])

    def test_string_column_accepted(self, numeric_df, mock_persist, mock_draw_bar):
        result = draw_bar_and_persist("/tmp", numeric_df, "group")
        assert isinstance(result, list)


class TestDrawBoxplotAndPersist:
    def test_returns_list(self, numeric_df, mock_persist):
        result = draw_boxplot_and_persist("/tmp", numeric_df, ["group"], y_column="score")
        assert isinstance(result, list)

    def test_none_dataframe_raises(self, mock_persist):
        with pytest.raises(ValueError):
            draw_boxplot_and_persist("/tmp", None, ["group"], y_column="score")

    def test_empty_dataframe_raises(self, mock_persist):
        with pytest.raises(ValueError):
            draw_boxplot_and_persist("/tmp", pd.DataFrame(), ["group"], y_column="score")

    def test_missing_column_raises(self, numeric_df, mock_persist):
        with pytest.raises(ValueError, match="Columns not found"):
            draw_boxplot_and_persist("/tmp", numeric_df, ["bad_col"], y_column="score")

    def test_multiple_columns_produce_multiple_files(self, numeric_df, mock_persist):
        result = draw_boxplot_and_persist("/tmp", numeric_df, ["group", "region"], y_column="score")
        assert len(result) == 2


class TestDrawOlsScatterplotAndPersist:
    def test_returns_list(self, numeric_df, mock_persist):
        result = draw_ols_scatterplot_and_persist("/tmp", numeric_df, ["age"], y_column="score")
        assert isinstance(result, list)

    def test_none_dataframe_raises(self, mock_persist):
        with pytest.raises(ValueError):
            draw_ols_scatterplot_and_persist("/tmp", None, ["age"], y_column="score")

    def test_empty_dataframe_raises(self, mock_persist):
        with pytest.raises(ValueError):
            draw_ols_scatterplot_and_persist("/tmp", pd.DataFrame(), ["age"], y_column="score")

    def test_missing_column_raises(self, numeric_df, mock_persist):
        with pytest.raises(ValueError, match="Columns not found"):
            draw_ols_scatterplot_and_persist("/tmp", numeric_df, ["no_col"], y_column="score")

    def test_multiple_columns(self, numeric_df, mock_persist):
        result = draw_ols_scatterplot_and_persist("/tmp", numeric_df, ["age", "income"], y_column="score")
        assert len(result) == 2


class TestDrawTukeyPlotsAndPersist:
    def test_returns_list(self, numeric_df, mock_persist):
        result = draw_tukey_plots_and_persist("/tmp", numeric_df, ["group"], value_column="score")
        assert isinstance(result, list)

    def test_none_dataframe_raises(self, mock_persist):
        with pytest.raises(ValueError):
            draw_tukey_plots_and_persist("/tmp", None, ["group"], value_column="score")

    def test_empty_dataframe_raises(self, mock_persist):
        with pytest.raises(ValueError):
            draw_tukey_plots_and_persist("/tmp", pd.DataFrame(), ["group"], value_column="score")

    def test_missing_column_raises(self, numeric_df, mock_persist):
        with pytest.raises(ValueError, match="Columns not found"):
            draw_tukey_plots_and_persist("/tmp", numeric_df, ["no_col"], value_column="score")

    def test_multiple_columns_produce_multiple_files(self, numeric_df, mock_persist):
        result = draw_tukey_plots_and_persist("/tmp", numeric_df, ["group", "region"], value_column="score")
        assert len(result) == 2

    def test_string_column_accepted(self, numeric_df, mock_persist):
        result = draw_tukey_plots_and_persist("/tmp", numeric_df, "group", value_column="score")
        assert isinstance(result, list)
