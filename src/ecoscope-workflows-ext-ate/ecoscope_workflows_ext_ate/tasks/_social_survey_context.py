import os 
import math
import jinja2
import warnings
import pandas as pd
from pathlib import Path
from docx.shared import Cm
from datetime import datetime
from docxtpl import DocxTemplate, InlineImage
from ecoscope_workflows_core.decorators import task
from ecoscope_workflows_core.tasks.filter._filter import TimeRange
from typing import Optional
from ecoscope_workflows_ext_custom.tasks.io._path_utils import remove_file_scheme

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff"}

warnings.filterwarnings("ignore")

def nan_to_empty(value):
    if (
        value is None
        or (isinstance(value, float) and math.isnan(value))
        or value == "nan"
    ):
        return ""
    print(value)
    return value

@task
def persist_survey_word(
    template_path: str,
    output_dir: str,
    time_period: Optional[TimeRange] = None,
    box_h_cm: float = 6.5,
    box_w_cm: float = 11.11,
    filename: str = "survey_report.docx",
    demographic_csv: str = "demographic_table.csv",
) -> str:
    """
    Render a docx template and inject images found in `output_dir` into the Jinja context.
    Image variable names = image filename stem without extension.
    e.g. output_dir/survey_locations_ecomap.png  -> context key 'survey_locations_ecomap'
    
    Also processes demographic_table.csv if present and adds to context as 'demographics'
    """
    jinja_env = jinja2.Environment()
    jinja_env.filters["nan_to_empty"] = nan_to_empty

    # Normalize paths
    template_path = remove_file_scheme(template_path)
    output_dir = remove_file_scheme(output_dir)

    print(f"\nTemplate Path: {template_path}")
    print(f"Output Directory: {output_dir}")

    if not template_path.strip():
        raise ValueError("template_path is empty after normalization")
    if not output_dir.strip():
        raise ValueError("output_directory is empty after normalization")
    if not os.path.exists(template_path):
        raise FileNotFoundError(f"Template file not found: {template_path}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    os.makedirs(output_dir, exist_ok=True)
    output_path = Path(output_dir) / filename
    print(f"Output File: {output_path}")

    time_period_str = None
    if time_period:
        fmt = getattr(time_period, "time_format", "%Y-%m-%d")
        time_period_str = f"{time_period.since.strftime(fmt)} to {time_period.until.strftime(fmt)}"
    
    base_context = {
        "report_period": time_period_str,
        "time_generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "prepared_by": "Ecoscope",
    }
    result = dict(base_context)

    # Load the template
    tpl = DocxTemplate(template_path)

    # ========================================================================
    # NEW: Process demographic CSV if it exists
    # ========================================================================
    demographic_path = Path(output_dir) / demographic_csv
    if demographic_path.exists():
        try:
            df = pd.read_csv(demographic_path)
            
            # Forward fill empty Demographic Variable cells
            df['Demographic Variable'] = df['Demographic Variable'].replace('', pd.NA).ffill()
            
            # Transform to nested structure for Jinja
            demographics = []
            for var_name, group in df.groupby('Demographic Variable', sort=False):
                categories = []
                for _, row in group.iterrows():
                    categories.append({
                        'category': row['Categories'] if row['Categories'] else 'Statistics',
                        'formatted_response': row['Number of responses']
                    })
                
                demographics.append({
                    'variable': var_name,
                    'categories': categories
                })
            
            # Add to context
            result['demographics'] = demographics

            # Calculate total responses from first variable's total
            try:
                # Extract n from first category of first variable
                first_response = demographics[0]['categories'][0]['formatted_response']
                import re
                match = re.search(r'n=(\d+)', first_response)
                if match:
                    # Sum all n values to get total
                    total = 0
                    for demo in demographics:
                        for cat in demo['categories']:
                            n_match = re.search(r'n=(\d+)', cat['formatted_response'])
                            if n_match:
                                total = max(total, int(n_match.group(1)))
                    result['total_responses'] = total
                else:
                    result['total_responses'] = None
            except:
                result['total_responses'] = None
                
        except Exception as e:
            print(f"Failed to process demographic CSV: {e}")
            result['demographics'] = []
    else:
        print(f"\nDemographic CSV not found: {demographic_path}")
        result['demographics'] = []

    # ========================================================================
    # Process images (existing code)
    # ========================================================================
    images_found = {}
    for root, _, files in os.walk(output_dir):
        for f in files:
            p = Path(root) / f
            if p.suffix.lower() in IMAGE_EXTS:
                var_name = p.stem
                images_found[var_name] = str(p)

    if images_found:
        for k, v in images_found.items():
            try:
                result[k] = InlineImage(tpl, v, width=Cm(box_w_cm), height=Cm(box_h_cm))
            except Exception as e:
                print(f"Failed to create InlineImage with fixed size for {v}: {e}. Trying without explicit height...")
                try:
                    result[k] = InlineImage(tpl, v, width=Cm(box_w_cm))
                except Exception as e2:
                    print(f"Failed to create InlineImage for {v}: {e2}. Skipping this image.")
    else:
        print("\nNo images found in output_dir.")

    # Check if string context values point to images
    for key, val in list(result.items()):
        if isinstance(val, str):
            p = Path(val)
            if p.exists() and p.suffix.lower() in IMAGE_EXTS:
                print(f"Context key '{key}' points to an image file; attaching as InlineImage: {p}")
                try:
                    result[key] = InlineImage(tpl, str(p), width=Cm(box_w_cm), height=Cm(box_h_cm))
                except Exception:
                    result[key] = InlineImage(tpl, str(p), width=Cm(box_w_cm))

    # ========================================================================
    # Render and save
    # ========================================================================
    try:
        tpl.render(result,jinja_env)
        tpl.save(output_path)
        return str(output_path)
    except Exception as e:
        print(f"\nError rendering document: {str(e)}")
        raise