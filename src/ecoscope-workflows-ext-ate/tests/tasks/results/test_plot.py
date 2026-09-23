"""Tests for ecoscope_workflows_ext_ate.tasks.results._plot.

`draw_likert_chart`, `draw_scatter_chart`, and `draw_tukey_chart` are
registered via `wt_registry.register()`, a no-op decorator at call time,
so they are called directly as plain Python against small, hand-built
DataFrames -- nothing here is mocked, since plotly/numpy/scipy are all
installed and cheap to run for real. Since each of these returns a
plotly `fig.to_html()` string rather than a structured figure, assertions
on them are necessarily string-level (type, presence of a `<div>`, whether
`widget_id` was embedded) -- the same style used for HTML-returning chart
functions elsewhere in this codebase.

The module-level `_infer_response_order`, `_generate_color_scheme`,
`_sort_dataframe`, and `_merge_layout` helpers (not registered) are also
covered directly, since they return plain Python data structures and hold
most of the interesting branching logic.
"""

from __future__ import annotations

import pandas as pd
import pytest

from ecoscope_workflows_ext_ate.tasks.results._plot import (
    LikertStyle,
    TrendlineStyle,
    _generate_color_scheme,
    _infer_response_order,
    _merge_layout,
    _sort_dataframe,
    draw_likert_chart,
    draw_scatter_chart,
    draw_tukey_chart,
)


# --------------------------------------------------------------------------- #
# _infer_response_order                                                       #
# --------------------------------------------------------------------------- #


class TestInferResponseOrder:
    def test_recognizes_a_known_five_point_agreement_scale_regardless_of_column_order(self):
        df = pd.DataFrame(
            {
                "q1": ["Strongly Disagree", "Agree", "Neutral"],
                "q2": ["Strongly Agree", "Disagree", "Neutral"],
            }
        )

        result = _infer_response_order(df)

        assert result == ["Strongly Disagree", "Disagree", "Neutral", "Agree", "Strongly Agree"]

    def test_recognizes_a_known_frequency_scale(self):
        df = pd.DataFrame({"q1": ["Never", "Always", "Sometimes"]})

        result = _infer_response_order(df)

        assert result == ["Never", "Sometimes", "Always"]

    def test_partial_pattern_match_only_returns_values_present_in_the_data(self):
        df = pd.DataFrame({"q1": ["Agree", "Strongly Agree"]})

        result = _infer_response_order(df)

        assert result == ["Agree", "Strongly Agree"]

    def test_unrecognized_values_fall_back_to_frequency_order(self):
        df = pd.DataFrame({"q1": ["Red", "Blue", "Red", "Green"]})

        result = _infer_response_order(df)

        # "Red" appears twice and so ranks first; ties keep encounter order.
        assert result == ["Red", "Blue", "Green"]


# --------------------------------------------------------------------------- #
# _generate_color_scheme                                                      #
# --------------------------------------------------------------------------- #


class TestGenerateColorScheme:
    def test_five_point_scale_uses_the_fixed_classic_palette(self):
        result = _generate_color_scheme(["A", "B", "C", "D", "E"])

        assert result == {
            "A": "#2c5282",
            "B": "#4299e1",
            "C": "#a0aec0",
            "D": "#ed8936",
            "E": "#c05621",
        }

    def test_seven_point_scale_uses_the_fixed_seven_step_palette(self):
        result = _generate_color_scheme(["A", "B", "C", "D", "E", "F", "G"])

        assert list(result.values()) == [
            "#1a365d",
            "#2c5282",
            "#4299e1",
            "#a0aec0",
            "#ed8936",
            "#c05621",
            "#9c4221",
        ]

    def test_non_five_or_seven_point_scale_still_returns_one_color_per_response(self):
        for n in (2, 3, 4, 6, 8):
            response_order = [f"r{i}" for i in range(n)]

            result = _generate_color_scheme(response_order)

            assert len(result) == n
            assert set(result.keys()) == set(response_order)
            assert all(v.startswith("#") and len(v) == 7 for v in result.values())


# --------------------------------------------------------------------------- #
# _sort_dataframe                                                             #
# --------------------------------------------------------------------------- #


class TestSortDataframe:
    def test_sort_by_name_orders_columns_alphabetically(self):
        df = pd.DataFrame({"b_question": [1], "a_question": [2]})

        result = _sort_dataframe(df, positive_responses=[], negative_responses=[], sort_by="name")

        assert list(result.columns) == ["a_question", "b_question"]

    def test_sort_by_positive_ranks_the_most_positive_column_first(self):
        df = pd.DataFrame(
            {
                "q_pos": ["Agree", "Agree", "Strongly Agree"],
                "q_neg": ["Disagree", "Disagree", "Strongly Disagree"],
            }
        )

        result = _sort_dataframe(
            df,
            positive_responses=["Agree", "Strongly Agree"],
            negative_responses=["Disagree", "Strongly Disagree"],
            sort_by="positive",
        )

        assert list(result.columns) == ["q_pos", "q_neg"]

    def test_sort_by_negative_ranks_the_most_negative_column_first(self):
        df = pd.DataFrame(
            {
                "q_pos": ["Agree", "Agree", "Strongly Agree"],
                "q_neg": ["Disagree", "Disagree", "Strongly Disagree"],
            }
        )

        result = _sort_dataframe(
            df,
            positive_responses=["Agree", "Strongly Agree"],
            negative_responses=["Disagree", "Strongly Disagree"],
            sort_by="negative",
        )

        assert list(result.columns) == ["q_neg", "q_pos"]

    def test_unknown_sort_by_leaves_original_column_order_unchanged(self):
        df = pd.DataFrame({"z_question": [1], "a_question": [2]})

        result = _sort_dataframe(df, positive_responses=[], negative_responses=[], sort_by="bogus")

        assert list(result.columns) == ["z_question", "a_question"]


# --------------------------------------------------------------------------- #
# _merge_layout                                                               #
# --------------------------------------------------------------------------- #


class TestMergeLayout:
    def test_top_level_override_replaces_the_default(self):
        result = _merge_layout({"title": "default"}, {"title": "custom"})

        assert result["title"] == "custom"

    def test_partial_xaxis_override_merges_instead_of_replacing(self):
        default = {"xaxis": {"tickmode": "array", "title": "default"}}
        override = {"xaxis": {"title": "custom"}}

        result = _merge_layout(default, override)

        assert result["xaxis"] == {"tickmode": "array", "title": "custom"}

    def test_axis_not_present_in_override_keeps_the_full_default(self):
        default = {"yaxis": {"tickmode": "array"}}

        result = _merge_layout(default, {})

        assert result["yaxis"] == {"tickmode": "array"}

    def test_keys_only_in_override_are_added(self):
        result = _merge_layout({"title": "default"}, {"height": 500})

        assert result == {"title": "default", "height": 500}


# --------------------------------------------------------------------------- #
# draw_likert_chart                                                           #
# --------------------------------------------------------------------------- #


@pytest.fixture
def likert_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "q1": ["Strongly Disagree", "Agree", "Neutral", "Strongly Agree", "Disagree"],
            "q2": ["Agree", "Agree", "Strongly Agree", "Neutral", "Disagree"],
        }
    )


class TestDrawLikertChart:
    def test_none_dataframe_raises_value_error(self):
        with pytest.raises(ValueError, match="None or empty"):
            draw_likert_chart(dataframe=None)

    def test_empty_dataframe_raises_value_error(self, likert_df):
        with pytest.raises(ValueError, match="None or empty"):
            draw_likert_chart(dataframe=likert_df.iloc[0:0])

    def test_returns_non_empty_html(self, likert_df):
        html = draw_likert_chart(dataframe=likert_df)

        assert isinstance(html, str)
        assert "<div" in html

    def test_widget_id_is_embedded_in_html(self, likert_df):
        html = draw_likert_chart(dataframe=likert_df, widget_id="my-likert-widget")

        assert "my-likert-widget" in html

    def test_explicit_response_order_and_style_overrides_do_not_raise(self, likert_df):
        html = draw_likert_chart(
            dataframe=likert_df,
            title="Custom Title",
            response_order=["Strongly Disagree", "Disagree", "Neutral", "Agree", "Strongly Agree"],
            likert_style=LikertStyle(sort_questions=True, sort_by="negative", show_percentages=False),
        )

        assert isinstance(html, str)
        assert "<div" in html


# --------------------------------------------------------------------------- #
# draw_scatter_chart                                                          #
# --------------------------------------------------------------------------- #


class TestDrawScatterChart:
    def test_returns_non_empty_html(self):
        df = pd.DataFrame({"x": [1, 2, 3, 4], "y": [2, 4, 6, 8]})

        html = draw_scatter_chart(dataframe=df, x_column="x", y_column="y")

        assert isinstance(html, str)
        assert "<div" in html

    def test_widget_id_is_embedded_in_html(self):
        df = pd.DataFrame({"x": [1, 2, 3], "y": [1, 2, 3]})

        html = draw_scatter_chart(dataframe=df, x_column="x", y_column="y", widget_id="my-scatter-widget")

        assert "my-scatter-widget" in html

    def test_all_null_pair_raises_value_error(self):
        df = pd.DataFrame({"x": [None, None], "y": [None, None]})

        with pytest.raises(ValueError, match="No valid data"):
            draw_scatter_chart(dataframe=df, x_column="x", y_column="y")

    def test_category_column_groups_points_into_separate_traces_without_raising(self):
        df = pd.DataFrame({"x": [1, 2, 3, 4], "y": [1, 2, 3, 4], "group": ["a", "a", "b", "b"]})

        html = draw_scatter_chart(dataframe=df, x_column="x", y_column="y", category_column="group")

        assert isinstance(html, str)
        assert "<div" in html

    def test_ols_trendline_does_not_raise(self):
        df = pd.DataFrame({"x": [1, 2, 3, 4, 5], "y": [2, 4, 5, 4, 5]})

        html = draw_scatter_chart(
            dataframe=df, x_column="x", y_column="y", trendline_style=TrendlineStyle(enabled=True, type="ols")
        )

        assert isinstance(html, str)
        assert "<div" in html

    def test_lowess_trendline_with_enough_points_does_not_raise(self):
        df = pd.DataFrame({"x": list(range(10)), "y": [v * 2 for v in range(10)]})

        html = draw_scatter_chart(
            dataframe=df, x_column="x", y_column="y", trendline_style=TrendlineStyle(enabled=True, type="lowess")
        )

        assert isinstance(html, str)
        assert "<div" in html

    def test_lowess_trendline_with_very_few_points_raises(self):
        # savgol_filter's window is forced to a minimum of 5, which exceeds
        # the number of available points here -- a real limitation of the
        # current implementation, not something this test works around.
        df = pd.DataFrame({"x": [1, 2, 3], "y": [1, 2, 3]})

        with pytest.raises(ValueError):
            draw_scatter_chart(
                dataframe=df, x_column="x", y_column="y", trendline_style=TrendlineStyle(enabled=True, type="lowess")
            )


# --------------------------------------------------------------------------- #
# draw_tukey_chart                                                            #
# --------------------------------------------------------------------------- #


@pytest.fixture
def tukey_comparisons_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "group1": ["A", "A", "B"],
            "group2": ["B", "C", "C"],
            "mean_diff": [1.0, 2.0, 3.0],
            "ci_lower": [0.5, 1.5, 2.5],
            "ci_upper": [1.5, 2.5, 3.5],
            "p_value": [0.01, 0.5, 0.02],
            "is_significant": [True, False, True],
        }
    )


class TestDrawTukeyChart:
    def test_none_comparisons_raises_value_error(self):
        with pytest.raises(ValueError, match="None or empty"):
            draw_tukey_chart(comparisons=None)

    def test_empty_comparisons_raises_value_error(self, tukey_comparisons_df):
        with pytest.raises(ValueError, match="None or empty"):
            draw_tukey_chart(comparisons=tukey_comparisons_df.iloc[0:0])

    def test_returns_non_empty_html(self, tukey_comparisons_df):
        html = draw_tukey_chart(comparisons=tukey_comparisons_df)

        assert isinstance(html, str)
        assert "<div" in html

    def test_widget_id_is_embedded_in_html(self, tukey_comparisons_df):
        html = draw_tukey_chart(comparisons=tukey_comparisons_df, widget_id="my-tukey-widget")

        assert "my-tukey-widget" in html
