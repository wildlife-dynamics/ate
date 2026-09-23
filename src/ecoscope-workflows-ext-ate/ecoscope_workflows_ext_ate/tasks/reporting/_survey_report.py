import os
import re
import math
import jinja2
import warnings
import pandas as pd
from pathlib import Path
from docx.shared import Cm
from typing import Optional
from datetime import datetime
from wt_registry import register
from docxtpl import DocxTemplate, InlineImage
from ecoscope.platform.tasks.filter._filter import TimeRange
from ecoscope_workflows_ext_custom.tasks.io._path_utils import remove_file_scheme

warnings.filterwarnings("ignore")

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff"}


def nan_to_empty(value):
    if value is None or (isinstance(value, float) and math.isnan(value)) or value == "nan":
        return ""
    return value


def _load_demographic_table(results_dir: Path | str, stem: str) -> Optional[pd.DataFrame]:
    """Look for `<stem>.parquet` first, then `<stem>.csv`, in results_dir."""
    results_dir = Path(results_dir)
    parquet_path = results_dir / f"{stem}.parquet"
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)

    csv_path = results_dir / f"{stem}.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)

    return None


def _build_demographics_context(df: pd.DataFrame) -> dict:
    df = df.copy()
    df["Demographic Variable"] = df["Demographic Variable"].replace("", pd.NA).ffill()

    demographics = []
    for var_name, group in df.groupby("Demographic Variable", sort=False):
        categories = [
            {
                "category": row["Categories"] if row["Categories"] else "Statistics",
                "formatted_response": row["Number of responses"],
            }
            for _, row in group.iterrows()
        ]
        demographics.append({"variable": var_name, "categories": categories})

    total_responses = None
    try:
        total = 0
        for demo in demographics:
            for cat in demo["categories"]:
                m = re.search(r"n=(\d+)", str(cat["formatted_response"]))
                if m:
                    total = max(total, int(m.group(1)))
        total_responses = total or None
    except Exception as e:
        print(f"Could not compute total_responses: {e}")

    return {"demographics": demographics, "total_responses": total_responses}


@register()
def generate_survey_report(
    template_path: str,
    output_dir: str,
    time_period: Optional[TimeRange] = None,
    prepared_by: str = "Ecoscope",
) -> str:
    """
    Render `template_path` (a docxtpl/Jinja Word template whose {{ variable }} tags are
    named after chart filename stems, e.g. {{ what_is_your_age_group_pie_chart }}) using
    every image found in `results_dir`, plus the demographic summary table, and save the
    rendered document to `output_path`.
    """
    template_path = remove_file_scheme(template_path)
    output_dir = remove_file_scheme(output_dir)

    jinja_env = jinja2.Environment()
    jinja_env.filters["nan_to_empty"] = nan_to_empty

    time_period_str = None
    if time_period:
        fmt = getattr(time_period, "time_format", "%Y-%m-%d")
        time_period_str = f"{time_period.since.strftime(fmt)} to {time_period.until.strftime(fmt)}"

    context = {
        "report_period": time_period_str,
        "time_generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "prepared_by": prepared_by,
    }

    tpl = DocxTemplate(str(template_path))

    demo_df = _load_demographic_table(output_dir, "demographic-data")
    if demo_df is not None:
        try:
            context.update(_build_demographics_context(demo_df))
        except Exception as e:
            print(f"Failed to process demographic table: {e}")
            context["demographics"] = []
    else:
        print(f"No demographic table found for stem 'demographic-data' in {output_dir}")
        context["demographics"] = []

    images_found = {}
    for root, _, files in os.walk(output_dir):
        for f in files:
            p = Path(root) / f
            if p.suffix.lower() in IMAGE_EXTS:
                images_found[p.stem] = str(p)

    if not images_found:
        print(f"No images found in {output_dir}")

    for var_name, image_path in images_found.items():
        try:
            context[var_name] = InlineImage(tpl, image_path, width=Cm(11.11), height=Cm(6.5))
        except Exception as e:
            print(f"Failed to embed {image_path} as {var_name!r}: {e}. Retrying without fixed height...")
            try:
                context[var_name] = InlineImage(tpl, image_path, width=Cm(11.11))
            except Exception as e2:
                print(f"Skipping {image_path}: {e2}")

    tpl.render(context, jinja_env)
    output_path = Path(output_dir) / "social_survey_report.docx"
    tpl.save(str(output_path))
    return str(output_path)
