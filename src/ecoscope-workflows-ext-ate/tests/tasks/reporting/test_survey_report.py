"""Tests for ecoscope_workflows_ext_ate.tasks.reporting._survey_report.

`generate_survey_report` is registered via `wt_registry.register()`, a
no-op decorator at call time, so it is called directly as plain Python. It
walks `output_dir` for chart images (matched to template placeholders by
filename stem) and an optional `demographic-data.parquet`/`.csv` table, so
tests exercise that against real files on disk (via `tmp_path`) and a real
.docx template (via python-docx/docxtpl), rather than mocking the
filesystem or docxtpl.

The module-level `nan_to_empty`, `_load_demographic_table`, and
`_build_demographics_context` helpers (not registered) are also covered
directly, since most of the interesting edge-case behavior lives there.
"""

from __future__ import annotations

import math
from pathlib import Path

import docx
import pandas as pd
import pytest
from docx.opc.exceptions import PackageNotFoundError
from ecoscope.platform.tasks.filter._filter import UTC_TIMEZONEINFO, TimeRange

from ecoscope_workflows_ext_ate.tasks.reporting._survey_report import (
    _build_demographics_context,
    _load_demographic_table,
    generate_survey_report,
    nan_to_empty,
)


# --------------------------------------------------------------------------- #
# nan_to_empty                                                                #
# --------------------------------------------------------------------------- #


class TestNanToEmpty:
    def test_none_becomes_empty_string(self):
        assert nan_to_empty(None) == ""

    def test_float_nan_becomes_empty_string(self):
        assert nan_to_empty(float("nan")) == ""

    def test_literal_string_nan_becomes_empty_string(self):
        assert nan_to_empty("nan") == ""

    def test_ordinary_string_passes_through_unchanged(self):
        assert nan_to_empty("Male") == "Male"

    def test_ordinary_number_passes_through_unchanged(self):
        assert nan_to_empty(5) == 5

    def test_zero_is_not_treated_as_nan(self):
        assert nan_to_empty(0) == 0


# --------------------------------------------------------------------------- #
# _load_demographic_table                                                     #
# --------------------------------------------------------------------------- #


class TestLoadDemographicTable:
    def test_returns_none_when_neither_file_exists(self, tmp_path):
        assert _load_demographic_table(tmp_path, "demographic-data") is None

    def test_loads_csv_when_only_csv_exists(self, tmp_path):
        pd.DataFrame({"a": [1]}).to_csv(tmp_path / "demographic-data.csv", index=False)

        result = _load_demographic_table(tmp_path, "demographic-data")

        assert result is not None
        assert result["a"].iloc[0] == 1

    def test_prefers_parquet_over_csv_when_both_exist(self, tmp_path):
        pd.DataFrame({"a": [1]}).to_csv(tmp_path / "demographic-data.csv", index=False)
        pd.DataFrame({"a": [2]}).to_parquet(tmp_path / "demographic-data.parquet")

        result = _load_demographic_table(tmp_path, "demographic-data")

        assert result["a"].iloc[0] == 2

    def test_accepts_a_string_path_as_well_as_a_path_object(self, tmp_path):
        pd.DataFrame({"a": [1]}).to_csv(tmp_path / "demographic-data.csv", index=False)

        result = _load_demographic_table(str(tmp_path), "demographic-data")

        assert result is not None


# --------------------------------------------------------------------------- #
# _build_demographics_context                                                 #
# --------------------------------------------------------------------------- #


class TestBuildDemographicsContext:
    def test_blank_demographic_variable_rows_are_forward_filled_into_groups(self):
        df = pd.DataFrame(
            {
                "Demographic Variable": ["Age", "", "Gender", ""],
                "Categories": ["18-25", "26-35", "M", "F"],
                "Number of responses": ["50.00% (n=5)", "50.00% (n=4)", "60.00% (n=6)", "40.00% (n=4)"],
            }
        )

        ctx = _build_demographics_context(df)

        variables = [d["variable"] for d in ctx["demographics"]]
        assert variables == ["Age", "Gender"]
        assert len(ctx["demographics"][0]["categories"]) == 2
        assert len(ctx["demographics"][1]["categories"]) == 2

    def test_blank_categories_are_labeled_statistics(self):
        df = pd.DataFrame(
            {
                "Demographic Variable": ["Age", ""],
                "Categories": ["18-25", ""],
                "Number of responses": ["50.00% (n=5)", "(mean=22.0)"],
            }
        )

        ctx = _build_demographics_context(df)

        categories = ctx["demographics"][0]["categories"]
        assert categories[1]["category"] == "Statistics"
        assert categories[1]["formatted_response"] == "(mean=22.0)"

    def test_total_responses_is_the_max_n_across_percentage_rows(self):
        df = pd.DataFrame(
            {
                "Demographic Variable": ["Gender", ""],
                "Categories": ["M", "F"],
                "Number of responses": ["60.00% (n=6)", "40.00% (n=4)"],
            }
        )

        ctx = _build_demographics_context(df)

        assert ctx["total_responses"] == 6

    def test_total_responses_is_none_when_no_n_pattern_is_present(self):
        df = pd.DataFrame({"Demographic Variable": ["Age"], "Categories": [""], "Number of responses": ["no data"]})

        ctx = _build_demographics_context(df)

        assert ctx["total_responses"] is None

    def test_total_responses_is_corrupted_by_stats_rows_containing_mean_or_median(self):
        # `_build_demographics_context` extracts total_responses via a
        # regex search for the literal substring "n=" followed by digits.
        # The words "mean=", "median=", and "min=" in the stats row
        # produced by format_demographic_table (e.g. "(mean=42.5; ...)")
        # all end in "n=" right before a number, so they are matched too --
        # and since the regex takes the max across every row, a stats row
        # can silently override the real respondent count. This documents
        # that real (surprising) behavior rather than the intended one.
        df = pd.DataFrame(
            {
                "Demographic Variable": ["Gender", "", "Age", ""],
                "Categories": ["M", "F", "18-25", ""],
                "Number of responses": [
                    "60.00% (n=6)",
                    "40.00% (n=4)",
                    "50.00% (n=5)",
                    "(mean=42.5; median=42; SD=15.14; max=65; min=20)",
                ],
            }
        )

        ctx = _build_demographics_context(df)

        # The true max respondent count here is 6, but the stats row's
        # "mean=42.5" is matched first and wins because 42 > 6.
        assert ctx["total_responses"] == 42


# --------------------------------------------------------------------------- #
# generate_survey_report                                                      #
# --------------------------------------------------------------------------- #


class TestGenerateSurveyReport:
    def test_saves_to_social_survey_report_docx_inside_output_dir(self, tmp_path, make_docx_template):
        template_path = make_docx_template(["{{ prepared_by }}"])

        result_path = generate_survey_report(template_path=str(template_path), output_dir=str(tmp_path))

        assert Path(result_path) == tmp_path / "social_survey_report.docx"
        assert Path(result_path).exists()

    def test_prepared_by_defaults_to_ecoscope(self, tmp_path, make_docx_template, read_docx_text):
        template_path = make_docx_template(["PREPARED:[{{ prepared_by }}]"])

        result_path = generate_survey_report(template_path=str(template_path), output_dir=str(tmp_path))

        texts = " ".join(read_docx_text(Path(result_path)))
        assert "PREPARED:[Ecoscope]" in texts

    def test_prepared_by_is_used_verbatim_when_given(self, tmp_path, make_docx_template, read_docx_text):
        template_path = make_docx_template(["PREPARED:[{{ prepared_by }}]"])

        result_path = generate_survey_report(
            template_path=str(template_path), output_dir=str(tmp_path), prepared_by="Jane Doe"
        )

        texts = " ".join(read_docx_text(Path(result_path)))
        assert "PREPARED:[Jane Doe]" in texts

    def test_time_period_is_formatted_using_its_own_time_format(self, tmp_path, make_docx_template, read_docx_text):
        template_path = make_docx_template(["PERIOD:[{{ report_period }}]"])
        time_period = TimeRange(
            since="2024-01-01", until="2024-01-31", timezone=UTC_TIMEZONEINFO, time_format="%Y-%m-%d"
        )

        result_path = generate_survey_report(
            template_path=str(template_path), output_dir=str(tmp_path), time_period=time_period
        )

        texts = " ".join(read_docx_text(Path(result_path)))
        assert "PERIOD:[2024-01-01 to 2024-01-31]" in texts

    def test_no_time_period_leaves_report_period_undefined(self, tmp_path, make_docx_template, read_docx_text):
        template_path = make_docx_template(["PERIOD:[{{ report_period }}]"])

        result_path = generate_survey_report(template_path=str(template_path), output_dir=str(tmp_path))

        texts = " ".join(read_docx_text(Path(result_path)))
        assert "PERIOD:[None]" in texts

    def test_chart_image_is_embedded_by_matching_its_filename_stem(
        self, tmp_path, make_png, make_docx_template, read_docx_text
    ):
        make_png(tmp_path / "what_is_your_age_group_pie_chart.png")
        template_path = make_docx_template(
            ["{% if what_is_your_age_group_pie_chart %}HAS_CHART{% endif %}"]
        )

        result_path = generate_survey_report(template_path=str(template_path), output_dir=str(tmp_path))

        texts = " ".join(read_docx_text(Path(result_path)))
        assert "HAS_CHART" in texts

    def test_charts_found_in_nested_subdirectories_are_still_picked_up(
        self, tmp_path, make_png, make_docx_template, read_docx_text
    ):
        make_png(tmp_path / "subdir" / "what_is_your_age_group_pie_chart.png")
        template_path = make_docx_template(
            ["{% if what_is_your_age_group_pie_chart %}HAS_CHART{% endif %}"]
        )

        result_path = generate_survey_report(template_path=str(template_path), output_dir=str(tmp_path))

        texts = " ".join(read_docx_text(Path(result_path)))
        assert "HAS_CHART" in texts

    def test_no_images_or_demographic_table_still_produces_a_document(self, tmp_path, make_docx_template):
        template_path = make_docx_template(["{{ prepared_by }}"])

        result_path = generate_survey_report(template_path=str(template_path), output_dir=str(tmp_path))

        assert Path(result_path).exists()

    def test_demographic_table_is_rendered_into_the_demographics_context(
        self, tmp_path, make_docx_template, read_docx_text
    ):
        pd.DataFrame(
            {
                "Demographic Variable": ["Gender", ""],
                "Categories": ["M", "F"],
                "Number of responses": ["60.00% (n=6)", "40.00% (n=4)"],
            }
        ).to_parquet(tmp_path / "demographic-data.parquet")
        template_path = make_docx_template(
            [
                "{% for demo in demographics %}VAR:{{ demo.variable }} "
                "{% for cat in demo.categories %}CAT:{{ cat.category }} {% endfor %}"
                "{% endfor %}",
                "TOTAL:[{{ total_responses }}]",
            ]
        )

        result_path = generate_survey_report(template_path=str(template_path), output_dir=str(tmp_path))

        texts = " ".join(read_docx_text(Path(result_path)))
        assert "VAR:Gender" in texts
        assert "CAT:M" in texts
        assert "CAT:F" in texts
        assert "TOTAL:[6]" in texts

    def test_nan_to_empty_filter_is_available_in_the_template(
        self, tmp_path, make_docx_template, read_docx_text
    ):
        pd.DataFrame(
            {
                "Demographic Variable": ["Gender", ""],
                "Categories": ["M", "F"],
                "Number of responses": ["60.00% (n=6)", float("nan")],
            }
        ).to_parquet(tmp_path / "demographic-data.parquet")
        template_path = make_docx_template(
            [
                "{% for demo in demographics %}{% for cat in demo.categories %}"
                "[{{ cat.formatted_response | nan_to_empty }}]"
                "{% endfor %}{% endfor %}"
            ]
        )

        result_path = generate_survey_report(template_path=str(template_path), output_dir=str(tmp_path))

        texts = " ".join(read_docx_text(Path(result_path)))
        assert "[60.00% (n=6)]" in texts
        assert "[]" in texts

    def test_no_demographic_table_falls_back_to_empty_demographics_list(
        self, tmp_path, make_docx_template, read_docx_text
    ):
        template_path = make_docx_template(
            ["{% if demographics %}HAS_DEMOGRAPHICS{% else %}NO_DEMOGRAPHICS{% endif %}"]
        )

        result_path = generate_survey_report(template_path=str(template_path), output_dir=str(tmp_path))

        texts = " ".join(read_docx_text(Path(result_path)))
        assert "NO_DEMOGRAPHICS" in texts

    def test_file_scheme_prefixed_paths_are_accepted(self, tmp_path, make_docx_template):
        template_path = make_docx_template(["{{ prepared_by }}"])

        result_path = generate_survey_report(
            template_path=f"file://{template_path}", output_dir=f"file://{tmp_path}"
        )

        assert Path(result_path).exists()

    def test_missing_template_path_raises(self, tmp_path):
        with pytest.raises(PackageNotFoundError):
            generate_survey_report(
                template_path=str(tmp_path / "missing_template.docx"), output_dir=str(tmp_path)
            )

    def test_output_dir_that_does_not_exist_yet_raises(self, tmp_path, make_docx_template):
        # Unlike the demographic-table/image lookups (which tolerate a
        # missing directory silently via os.walk), the final tpl.save()
        # call requires output_dir to already exist -- there is no
        # mkdir(parents=True) anywhere in this function.
        template_path = make_docx_template(["{{ prepared_by }}"])
        missing_dir = tmp_path / "does_not_exist_yet"

        with pytest.raises(FileNotFoundError):
            generate_survey_report(template_path=str(template_path), output_dir=str(missing_dir))
