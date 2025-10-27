import os 
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
import statsmodels.api as sm   
import plotly.graph_objects as go
from typing_extensions import Literal
from statsmodels.formula.api import ols
from ecoscope.base.utils import hex_to_rgba
from ecoscope_workflows_core.decorators import task
from ecoscope_workflows_core.tasks.io import persist_text
from ecoscope_workflows_core.annotations import AnyDataFrame
from typing import Iterable, Optional, Union,List,Dict,Annotated,cast
from ecoscope_workflows_core.tasks.transformation._mapping import map_values
from ecoscope_workflows_ext_ecoscope.tasks.results._ecoplot import ExportArgs
from ecoscope_workflows_ext_ecoscope.tasks.results._ecoplot import (
    draw_pie_chart, PlotStyle, 
    LayoutStyle,draw_bar_chart,
    BarConfig,PlotCategoryStyle
)

from pydantic import Field,BaseModel
from pydantic.json_schema import SkipJsonSchema
from ecoscope_workflows_core.annotations import (
    AdvancedField,
    DataFrame,
    JsonSerializableDataFrameModel,
)

warnings.filterwarnings("ignore")
class TukeyPlotStyle(BaseModel):
    significant_color: Annotated[
        str | SkipJsonSchema[None], AdvancedField(default="#ff6b6b")
    ] = "#ff6b6b"
    non_significant_color: Annotated[
        str | SkipJsonSchema[None], AdvancedField(default="#4ecdc4")
    ] = "#4ecdc4"
    marker_size: Annotated[int | SkipJsonSchema[None], AdvancedField(default=8)] = 8
    line_width: Annotated[int | SkipJsonSchema[None], AdvancedField(default=2)] = 2
    confidence_level: Annotated[
        float | SkipJsonSchema[None], AdvancedField(ge=0.0, le=1.0, default=0.95)
    ] = 0.95


class ScatterStyle(BaseModel):
    marker_size: Annotated[
        int | SkipJsonSchema[None], AdvancedField(default=None)
    ] = None
    marker_color: Annotated[
        str | SkipJsonSchema[None], AdvancedField(default=None)
    ] = None
    marker_symbol: Annotated[
        str | SkipJsonSchema[None], AdvancedField(default=None)
    ] = None
    marker_opacity: Annotated[
        float | SkipJsonSchema[None], AdvancedField(ge=0.0, le=1.0, default=None)
    ] = None
    mode: Annotated[
        str | SkipJsonSchema[None], AdvancedField(default="markers")
    ] = "markers"


class TrendlineStyle(BaseModel):
    enabled: Annotated[bool, AdvancedField(default=False)] = False
    type: Annotated[
        Literal["ols", "lowess"] | SkipJsonSchema[None],
        AdvancedField(
            default="ols",
            description="Type of trendline. 'ols' for linear regression, 'lowess' for locally weighted  smoothing.",
        ),
    ] = "ols"
    color: Annotated[
        str | SkipJsonSchema[None], AdvancedField(default="red")
    ] = "red"
    width: Annotated[int | SkipJsonSchema[None], AdvancedField(default=2)] = 2
    dash: Annotated[
        str | SkipJsonSchema[None], AdvancedField(default="solid")
    ] = "solid"



def _normalize_columns(df: AnyDataFrame, columns: Optional[Union[str, Iterable[str]]]):
    if columns is None:
        # default: all object dtype cols
        return list(df.select_dtypes(include="object").columns)
    if isinstance(columns, str):
        return [columns]
    return list(columns)

@task
def convert_object_to_value(df: AnyDataFrame, columns: Optional[Union[str, Iterable[str]]] = None) -> AnyDataFrame:
    """
    Convert given object columns to numeric (float/int where possible).
    Non-convertible values become NaN (errors='coerce').
    Modifies df in-place and returns it.
    """
    if df is None or not isinstance(df, pd.DataFrame):
        raise ValueError("df must be a pandas DataFrame")

    cols = _normalize_columns(df, columns)
    for col in cols:
        if col not in df.columns:
            raise ValueError(f"Column not found: {col}")
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

@task
def convert_object_to_string(df: AnyDataFrame, columns: Optional[Union[str, Iterable[str]]] = None) -> AnyDataFrame:
    """
    Convert given object columns to string dtype.
    Modifies df in-place and returns it.
    """
    if df is None or not isinstance(df, pd.DataFrame):
        raise ValueError("df must be a pandas DataFrame")
    
    cols = _normalize_columns(df, columns)
    for col in cols:
        if col not in df.columns:
            raise ValueError(f"Column not found: {col}")
        df[col] = df[col].astype("string")  # pandas string dtype, better than plain Python str
    
    return df

@task
def format_demographic_table(
    df: AnyDataFrame,
    columns_of_interest: list,
) -> AnyDataFrame:
    """
    Return a tidy DataFrame summarizing categorical and numeric demographic columns.
    - Categorical columns: value counts (percentage and n).
    - Numeric columns: binned counts plus a stats row (mean, median, sd, min, max).
    Automatically detects numeric columns and creates appropriate bins.
    """
    
    def create_bins(series: pd.Series, n_bins: int = 5):
        """Create bins for a numeric series."""
        clean_data = series.dropna()
        if clean_data.empty:
            return None, None
        
        min_val = clean_data.min()
        max_val = clean_data.max()
        
        if min_val == max_val:
            return [min_val - 1, max_val + 1], [f"{min_val}"]
        
        try:
            _, bin_edges = pd.qcut(clean_data, q=n_bins, retbins=True, duplicates='drop')
            bin_edges[0] = 0  # Start from 0
            bin_edges[-1] = float('inf')  # Extend to infinity
            
            # Create labels
            labels = []
            for i in range(len(bin_edges) - 1):
                if bin_edges[i+1] == float('inf'):
                    labels.append(f"{int(bin_edges[i])}+")
                else:
                    labels.append(f"{int(bin_edges[i])}-{int(bin_edges[i+1])}")
            
            return bin_edges.tolist(), labels
        except:
            bin_edges = np.linspace(0, max_val * 1.1, n_bins + 1)
            bin_edges[-1] = float('inf')
            labels = [f"{int(bin_edges[i])}-{int(bin_edges[i+1]) if bin_edges[i+1] != float('inf') else '+'}" 
                     for i in range(len(bin_edges) - 1)]
            return bin_edges.tolist(), labels
    
    rows = []
    total = len(df)
    
    for col in columns_of_interest:
        if col not in df.columns:
            continue
            
        numeric = pd.to_numeric(df[col], errors='coerce')
        if numeric.notna().sum() > 0 and numeric.notna().sum() / len(numeric) > 0.5:
            bins, labels = create_bins(numeric)
            
            if bins and labels:
                binned = pd.cut(numeric, bins=bins, labels=labels)
                counts = binned.value_counts().reindex(labels, fill_value=0)
                
                for cat, cnt in counts.items():
                    pct = (cnt / total) * 100 if total else 0
                    rows.append({
                        "Demographic Variable": col,
                        "Categories": str(cat),
                        "Number of responses": f"{pct:.2f}% (n={int(cnt)})"
                    })
                
                stats = numeric.dropna()
                if not stats.empty:
                    stats_text = (
                        f"(mean={stats.mean():.1f}; median={stats.median():.0f}; "
                        f"SD={stats.std():.2f}; max={stats.max():.0f}; min={stats.min():.0f})"
                    )
                else:
                    stats_text = "(no numeric data)"
                    
                rows.append({
                    "Demographic Variable": "",
                    "Categories": "",
                    "Number of responses": stats_text
                })
        else:
            series = df[col].fillna("No Response").astype(str)
            counts = series.value_counts(dropna=False)
            
            for cat, cnt in counts.items():
                pct = (cnt / total) * 100 if total else 0
                rows.append({
                    "Demographic Variable": col,
                    "Categories": str(cat),
                    "Number of responses": f"{pct:.2f}% (n={int(cnt)})"
                })
    
    result_df = pd.DataFrame(rows)
    if not result_df.empty:
        out_rows = []
        current_var = None
        for _, r in result_df.iterrows():
            if r["Demographic Variable"] == current_var:
                r = r.copy()
                r["Demographic Variable"] = ""
            else:
                current_var = r["Demographic Variable"]
            out_rows.append(r)
        result_df = pd.DataFrame(out_rows)
    
    return result_df

def create_bins(series: pd.Series, bins: int = 5, min_start: int = 0) -> pd.Series:
    """
    Create whole-number bins for a numeric pandas Series.
    - Forces values <= 0 to NaN (excluded from bins).
    - Builds integer edges so labels are unique.
    - If rounding collapses edges, reduces number of bins automatically.
    - Returns string labels like "1–5" and None for missing values.
    """
    s = pd.to_numeric(series, errors="coerce")
    s_pos = s.where(s > 0, np.nan)
    if s_pos.dropna().empty:
        return pd.Series([None] * len(s_pos), index=s_pos.index, dtype=object)

    raw_min = int(np.floor(s_pos.min()))
    raw_max = int(np.ceil(s_pos.max()))
    start = max(min_start, raw_min)
    end = max(start + 1, raw_max)
    desired_edges = np.linspace(start, end, bins + 1)
    int_edges = np.unique(np.round(desired_edges).astype(int))
    if int_edges.size < 2:
        int_edges = np.array([start, end], dtype=int)
    effective_bins = int_edges.size - 1
    if effective_bins <= 0:
        int_edges = np.array([start, start + 1], dtype=int)
        effective_bins = 1

    cat = pd.cut(s_pos, bins=int_edges, include_lowest=True)

    labels = []
    for left, right in zip(int_edges[:-1], int_edges[1:]):
        r_display = max(right, left + 1)
        labels.append(f"{int(left)}–{int(r_display)}")

    if len(labels) == len(cat.cat.categories):
        cat = cat.cat.rename_categories(labels)
    else:
        cat = cat.astype(object).where(~cat.isna(), other=None)
        return cat
    out = cat.astype(object).where(~cat.isna(), other=None)
    out.index = s_pos.index
    return out

@task
def create_likert_chart(
    df: AnyDataFrame,
    title: str = None,
    response_order: Optional[List[str]] = None,
    colors: Optional[Dict[str, str]] = None,
    neutral_categories: Optional[List[str]] = None,
    height_per_question: int = 60,
    min_height: int = 400,
    width: int = 1200,
    show_percentages: bool = True,
    sort_questions: bool = False,
    sort_by: str = 'positive'  # 'positive', 'negative', or 'name'
) -> str:
    """
    Create a generalized Likert scale chart (diverging stacked bar chart).
    
    Args:
        df: DataFrame where columns are questions and values are response categories
        title: Chart title
        response_order: Order of responses from most negative to most positive.
                       If None, inferred from data
        colors: Dictionary mapping response categories to colors.
                If None, uses default color scheme
        neutral_categories: List of categories considered neutral (displayed on positive side).
                           If None, auto-detects or uses middle category
        height_per_question: Height in pixels per question/row
        min_height: Minimum chart height
        width: Chart width
        show_percentages: Whether to show percentage labels on bars
        sort_questions: Whether to sort questions by response distribution
        sort_by: How to sort questions ('positive', 'negative', 'name')
    
    Returns:
        Plotly Figure object
    """
    if response_order is None:
        response_order = _infer_response_order(df)
    
    if colors is None:
        colors = _generate_color_scheme(response_order)
    
    mid_point = len(response_order) // 2
    
    if neutral_categories is None:
        if len(response_order) % 2 == 1:
            neutral_categories = [response_order[mid_point]]
            negative_responses = response_order[:mid_point]
            positive_responses = response_order[mid_point + 1:]
        else:
            neutral_categories = []
            negative_responses = response_order[:mid_point]
            positive_responses = response_order[mid_point:]
    else:
        negative_responses = [r for r in response_order if r not in neutral_categories 
                              and response_order.index(r) < mid_point]
        positive_responses = [r for r in response_order if r not in neutral_categories 
                             and response_order.index(r) >= mid_point]
    
    if sort_questions:
        df = _sort_dataframe(df, response_order, positive_responses, 
                            negative_responses, sort_by)
    
    percentages_df = df.apply(lambda col: col.value_counts() / len(col) * 100, axis=0)
    fig = go.Figure()
    legend_rank = len(response_order)
    for response in reversed(negative_responses):
        values = []
        for col in df.columns:
            percentage = percentages_df.loc[response, col] if response in percentages_df.index else 0
            values.append(-percentage)
        
        fig.add_trace(
            go.Bar(
                x=values,
                y=df.columns,
                orientation="h",
                name=response,
                marker_color=colors.get(response, "#808080"),
                customdata=np.abs(values),
                hovertemplate="%{y}<br>%{fullData.name}: %{customdata:.1f}%<extra></extra>",
                text=[f"{abs(v):.1f}%" if show_percentages and abs(v) > 5 else "" for v in values],
                textposition="inside",
                textfont=dict(size=11, color="white"),
                legendrank=legend_rank,
            )
        )
        legend_rank -= 1
    
    for response in neutral_categories + positive_responses:
        values = []
        for col in df.columns:
            percentage = percentages_df.loc[response, col] if response in percentages_df.index else 0
            values.append(percentage)
        
        fig.add_trace(
            go.Bar(
                x=values,
                y=df.columns,
                orientation="h",
                name=response,
                marker_color=colors.get(response, "#808080"),
                customdata=values,
                hovertemplate="%{y}<br>%{fullData.name}: %{customdata:.1f}%<extra></extra>",
                text=[f"{v:.1f}%" if show_percentages and v > 5 else "" for v in values],
                textposition="inside",
                textfont=dict(size=11, color="white"),
                legendrank=legend_rank,
            )
        )
        legend_rank -= 1
    chart_height = max(min_height, len(df.columns) * height_per_question)
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=24, color="black", family="Arial"),
            x=0.5,
            xanchor="center",
            y=0.98,
            yanchor="top",
        ),
        barmode="relative",
        height=chart_height,
        width=width,
        yaxis_autorange="reversed",
        bargap=0.15,
        margin=dict(l=300, r=100, t=150, b=80),
        xaxis=dict(
            title="Percentage of Responses",
            title_font=dict(size=14, family="Arial"),
            range=[-100, 100],
            ticksuffix="%",
            zeroline=True,
            zerolinecolor="black",
            zerolinewidth=2,
            gridcolor="#e2e8f0",
            showline=True,
            linecolor="black",
            tickfont=dict(size=12, family="Arial"),
        ),
        yaxis=dict(
            tickfont=dict(size=12, color="black", family="Arial"),
            linecolor="black",
            showline=True,
            gridcolor="#e2e8f0",
        ),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(size=12, family="Arial"),
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="black",
            borderwidth=1,
            traceorder="normal",
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    
    # Add border
    fig.update_layout(
        shapes=[
            dict(
                type="rect",
                xref="paper",
                yref="paper",
                x0=0,
                y0=0,
                x1=1,
                y1=1,
                line=dict(color="black", width=1),
            )
        ]
    )
    
    return fig.to_html(**ExportArgs().model_dump())

def _infer_response_order(df: AnyDataFrame) -> List[str]:
    """Infer response order from data."""
    common_patterns = [
        ["Strongly Disagree", "Disagree", "Neutral", "Agree", "Strongly Agree"],
        ["Very Dissatisfied", "Dissatisfied", "Neutral", "Satisfied", "Very Satisfied"],
        ["Never", "Rarely", "Sometimes", "Often", "Always"],
        ["Very Unlikely", "Unlikely", "Neutral", "Likely", "Very Likely"],
        ["Very Poor", "Poor", "Fair", "Good", "Excellent"],
        ["Strongly Oppose", "Oppose", "Neutral", "Support", "Strongly Support"],
    ]
    all_values = set()
    for col in df.columns:
        all_values.update(df[col].dropna().unique())
    
    for pattern in common_patterns:
        if all_values.issubset(set(pattern)):
            return [p for p in pattern if p in all_values]
    
    value_counts = pd.Series([v for col in df.columns for v in df[col].dropna()]).value_counts()
    return value_counts.index.tolist()


def _generate_color_scheme(response_order: List[str]) -> Dict[str, str]:
    """Generate color scheme based on number of responses."""
    n = len(response_order)
    
    if n == 5:
        # Classic 5-point Likert
        return {
            response_order[0]: "#2c5282",  # Dark blue
            response_order[1]: "#4299e1",  # Light blue
            response_order[2]: "#a0aec0",  # Grey
            response_order[3]: "#ed8936",  # Orange
            response_order[4]: "#c05621",  # Dark orange
        }
    elif n == 7:
        # 7-point scale
        colors_list = ["#1a365d", "#2c5282", "#4299e1", "#a0aec0", "#ed8936", "#c05621", "#9c4221"]
        return dict(zip(response_order, colors_list))
    else:
        import colorsys
        colors = []
        mid = n // 2
        
        for i in range(n):
            if i < mid:
                ratio = i / mid if mid > 0 else 0
                h, s, v = 0.6, 0.8 - (ratio * 0.3), 0.5 + (ratio * 0.2)
            elif i == mid and n % 2 == 1:
                h, s, v = 0, 0, 0.65
            else:
                ratio = (i - mid) / (n - mid) if (n - mid) > 0 else 0
                h, s, v = 0.08, 0.6 + (ratio * 0.3), 0.8 - (ratio * 0.2)
            
            rgb = colorsys.hsv_to_rgb(h, s, v)
            colors.append(f"#{int(rgb[0]*255):02x}{int(rgb[1]*255):02x}{int(rgb[2]*255):02x}")
        return dict(zip(response_order, colors))

def _sort_dataframe(
    df: AnyDataFrame,
    response_order: List[str],
    positive_responses: List[str],
    negative_responses: List[str],
    sort_by: str
) -> AnyDataFrame:
    if sort_by == 'name':
        return df[sorted(df.columns)]
    scores = {}
    for col in df.columns:
        counts = df[col].value_counts()
        total = counts.sum()
        
        if sort_by == 'positive':
            score = sum(counts.get(r, 0) for r in positive_responses) / total if total > 0 else 0
        elif sort_by == 'negative':
            score = sum(counts.get(r, 0) for r in negative_responses) / total if total > 0 else 0
        else:
            score = 0
        
        scores[col] = score
    sorted_cols = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    return df[sorted_cols]


@task
def map_survey_responses(
    df: AnyDataFrame,
    columns: List[str],
    value_map: Dict[str, str],
    inplace: bool = False
) -> AnyDataFrame:
    if not inplace:
        df = df.copy()
    for column in columns:
        if column in df.columns:
            df[column] = df[column].map(value_map).fillna(df[column])
        else:
            print(f"Warning: Column '{column}' not found in DataFrame. Skipping.")
    return df

@task
def convert_to_int(
    df: AnyDataFrame,
    columns: Union[str, List[str]],
    errors: str = 'coerce',
    fill_value: int = 0,
    inplace: bool = False
) -> AnyDataFrame:
    if not inplace:
        df = df.copy()
    if isinstance(columns, str):
        columns = [columns]
    
    for column in columns:
        if column not in df.columns:
            print(f"Warning: Column '{column}' not found in DataFrame. Skipping.")
            continue
        
        try:
            if errors == 'coerce':
                df[column] = pd.to_numeric(df[column], errors='coerce').fillna(fill_value).astype(int)
            else:
                df[column] = df[column].astype(int)
        except Exception as e:
            print(f"Error converting column '{column}' to int: {e}")
            if errors == 'raise':
                raise
    
    return df

@task
def bin_columns(
    df: AnyDataFrame, 
    columns: Union[str, List[str]], 
    bins: int = 5,
    suffix: str = " bins",
    inplace: bool = False
) -> AnyDataFrame:
    """
    Create binned versions of numeric columns.
    
    Args:
        df: Input DataFrame
        columns: Column name (str) or list of column names to bin
        bins: Number of bins to create (default 5)
        suffix: Suffix to add to new column names (default " bins")
        inplace: Whether to modify DataFrame in place
    
    Returns:
        DataFrame with new binned columns
    """
    if not inplace:
        df = df.copy()
    
    # Convert single column to list
    if isinstance(columns, str):
        columns = [columns]
    
    for column in columns:
        if column not in df.columns:
            print(f"Warning: Column '{column}' not found in DataFrame. Skipping.")
            continue
        
        try:
            new_col = f"{column}{suffix}"
            df[new_col] = create_bins(df[column], bins=bins)
        except Exception as e:
            print(f"Error binning column '{column}': {e}")
            continue
    
    return df

@task
def get_chart_recommendations(
    df: AnyDataFrame, 
    exclude_columns: Optional[List[str]] = None,
    max_categorical_unique: int = 20,
    pie_chart_threshold: int = 7
) -> AnyDataFrame:
    chart_recommendations = {}
    df_analysis = df.copy()
    if exclude_columns:
        df_analysis = df_analysis.drop(columns=exclude_columns, errors="ignore")
    
    for column in df_analysis.columns:
        unique_count = df_analysis[column].nunique()
        dtype = df_analysis[column].dtype
        if df_analysis[column].isna().all():
            continue
        if dtype == "object" or dtype.name == "category":
            if unique_count > max_categorical_unique:
                chart_type = "Not Recommended"
                reason = f"Too many unique values ({unique_count})"
            elif unique_count <= pie_chart_threshold:
                chart_type = "Pie Chart"
                reason = "Low cardinality"
            else:
                chart_type = "Bar Chart"
                reason = "Moderate cardinality"

        elif dtype == "bool":
            chart_type = "Pie Chart"
            reason = "Binary data"
        elif dtype in ["int64", "float64", "int32", "float32"]:
            if unique_count <= 10:
                chart_type = "Bar Chart"
                reason = "Discrete numeric with few values"
            else:
                sorted_data = df_analysis[column].dropna().sort_values()
                if len(sorted_data) > 1:
                    diff_unique = sorted_data.diff().nunique()
                    if diff_unique <= 3 and sorted_data.is_monotonic_increasing:
                        chart_type = "Line Chart"
                        reason = "Sequential/temporal pattern"
                    else:
                        chart_type = "Histogram"
                        reason = "Continuous numeric distribution"
                else:
                    chart_type = "Bar Chart"
                    reason = "Insufficient data"
        elif pd.api.types.is_datetime64_any_dtype(df_analysis[column]):
            chart_type = "Line Chart"
            reason = "Temporal data"
        
        else:
            chart_type = "Bar Chart"
            reason = f"Default for {dtype}"
        
        chart_recommendations[column] = {
            "chart_type": chart_type,
            "unique_count": unique_count,
            "data_type": str(dtype),
            "reason": reason
        }
    
    chart_df = pd.DataFrame.from_dict(chart_recommendations, orient='index').reset_index()
    chart_df.columns = ["column", "chart_type", "unique_count", "data_type", "reason"]
    
    return chart_df

@task 
def filter_cols_df(
    df: AnyDataFrame, 
    cols: Union[str, List[str]],
    errors: str = 'raise'
) -> AnyDataFrame:
    """
    Filter DataFrame to include only specified columns.
    
    Args:
        df: Input DataFrame
        cols: Column name (str) or list of column names to keep
        errors: How to handle missing columns ('raise', 'ignore', 'warn')
                'raise' - raises KeyError if columns not found
                'ignore' - silently skips missing columns
                'warn' - prints warning and skips missing columns
    
    Returns:
        DataFrame with only the specified columns
    """
    df_copy = df.copy()
    if isinstance(cols, str):
        cols = [cols]
    if errors == 'raise':
        return df_copy[cols]
    
    elif errors == 'ignore':
        valid_cols = [col for col in cols if col in df_copy.columns]
        return df_copy[valid_cols]
    
    elif errors == 'warn':
        valid_cols = []
        for col in cols:
            if col in df_copy.columns:
                valid_cols.append(col)
            else:
                print(f"Warning: Column '{col}' not found in DataFrame. Skipping.")
        return df_copy[valid_cols]
    
    else:
        raise ValueError(f"errors must be 'raise', 'ignore', or 'warn', got '{errors}'")

@task
def draw_pie_and_persist(output_dir, df: AnyDataFrame, columns: Union[str, List[str]]):
    if df is None or df.empty:
        raise ValueError("DataFrame is None or empty")
    if isinstance(columns, str):
        columns = [columns]
    elif not isinstance(columns, list):
        raise ValueError("columns parameter must be a string or list of strings")
    missing_cols = [col for col in columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Columns not found in dataframe: {missing_cols}")
    
    pie_dir_list = []
    cat_options = {
        "Yes": "#a8e6cf",  # Pastel green
        "No": "#ffb3ba",  # Pastel red
        "I don't know": "#bae1ff",  # Pastel blue
        "Prefer not to answer": "#e0e0e0",  # Pastel gray
        "False": "#ffd4a3",  # Pastel orange
        "True": "#c7ceea",  # Pastel purple
        "Unspecified": "#f5f5dc",  # Pastel beige
        "Strongly agree": "#b5ead7",  # Pastel mint
        "Agree": "#c7f0bd",  # Pastel lime
        "Neutral": "#fff5ba",  # Pastel yellow
        "Disagree": "#ffd3b6",  # Pastel peach
        "Strongly disagree": "#ffaaa5",  # Pastel coral
    }
    
    default_color = '#5f9ea0'
    default_rgba = hex_to_rgba(default_color)
    for column in columns:
        try:
            df_filtered = df[df[column].notna()].copy()
            if df_filtered.empty:
                print(f"Skipping column `{column}`: no valid data after removing NaN values")
                continue
            
            unique_values = df_filtered[column].unique()
            column_colors = []
            for value in unique_values:
                if value in cat_options:
                    column_colors.append(cat_options[value])
                else:
                    column_colors.append(default_color)
            df_filtered['hex_colors'] = df_filtered[column].map(cat_options)
            df_filtered['hex_colors'].fillna(default_color, inplace=True)
            df_filtered["colors"] = df_filtered["hex_colors"].apply(hex_to_rgba)
            
            pie_chart = draw_pie_chart(
                dataframe=df_filtered,
                value_column=column,
                label_column=column,
                color_column="colors",
                plot_style=PlotStyle(
                    textinfo="percent", 
                    marker_colors=column_colors
                ),
                layout_style=LayoutStyle(
                    font_size=9,
                    font_style="normal",
                    showlegend=True,
                ),
            )
            
            new_col = column.lower().replace(" ", "_")
            
            if pie_chart is None:
                print(f"An error occurred when generating `{column}` pie chart")
            else:
                file_path = persist_text(
                    pie_chart, 
                    output_dir, 
                    f"{new_col}_pie_chart.html"
                )
                print(f"Successfully saved pie chart for `{column}` to {file_path}")
                pie_dir_list.append(file_path)
                
        except Exception as e:
            print(f"Error processing column `{column}`: {str(e)}")
            continue
            
    return pie_dir_list

@task
def draw_bar_and_persist(output_dir, df: AnyDataFrame, columns: Union[str, List[str]]):
    if df is None or df.empty:
        raise ValueError("DataFrame is None or empty")

    if isinstance(columns, str):
        columns = [columns]
    elif not isinstance(columns, list):
        raise ValueError("columns parameter must be a string or list of strings")
    
    missing_cols = [col for col in columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Columns not found in dataframe: {missing_cols}")
    
    bar_dir_list = []
    bar_configs = [
        BarConfig(
            column="id",                    
            agg_func="count",   
            label="",    
            style=PlotCategoryStyle(marker_color="#7eb0d5") 
        )
     ]

    for column in columns:
        try:
            
            bar_chart = draw_bar_chart(
                dataframe=df,
                bar_chart_configs=bar_configs,
                category=column,
                layout_kwargs=LayoutStyle(
                    font_size = 13,
                    font_color="#222222",
                    title_x=0.5,
                    xaxis={'title': ""},
                    yaxis={'title': 'Count'},
                    showlegend=False,
                    bargap=0.1,
                )
            )
            new_col = column.lower().replace(" ", "_")
            if bar_chart is None:
                print(f"An error occurred when generating `{column}` bar chart")
            else:
                file_path = persist_text(
                    bar_chart, 
                    output_dir, 
                    f"{new_col}_bar_chart.html"
                )
                print(f"Successfully saved pie chart for `{column}` to {file_path}")
                bar_dir_list.append(file_path)
                
        except Exception as e:
            print(f"Error processing column `{column}`: {str(e)}")
            continue  
    return bar_dir_list

@task
def fill_missing_values(
    dataframe: AnyDataFrame,
    numeric_columns: Annotated[
        list[str] | SkipJsonSchema[None],
        Field(
            default=None,
            description="List of numeric column names to fill missing values for.",
        ),
    ] = None,
    numeric_fill_value: Annotated[
        float | int | SkipJsonSchema[None],
        Field(
            default=0,
            description="Value to use for filling missing numeric values.",
        ),
    ] = 0,
    categorical_columns: Annotated[
        list[str] | SkipJsonSchema[None],
        Field(
            default=None,
            description="List of categorical column names to fill missing values for.",
        ),
    ] = None,
    categorical_fill_value: Annotated[
        str | SkipJsonSchema[None],
        Field(
            default="Unspecified",
            description="Value to use for filling missing categorical values.",
        ),
    ] = "Unspecified",
) -> AnyDataFrame:
    import pandas as pd
    df_filled = dataframe.copy()
    if numeric_columns is None:
        numeric_columns = df_filled.select_dtypes(include=["number"]).columns.tolist()
    
    if categorical_columns is None:
        categorical_columns = df_filled.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()
    missing_numeric = [col for col in numeric_columns if col not in df_filled.columns]
    if missing_numeric:
        raise ValueError(f"Numeric columns not found in dataframe: {missing_numeric}")
    
    missing_categorical = [
        col for col in categorical_columns if col not in df_filled.columns
    ]
    if missing_categorical:
        raise ValueError(
            f"Categorical columns not found in dataframe: {missing_categorical}"
        )
    if numeric_columns:
        df_filled[numeric_columns] = df_filled[numeric_columns].fillna(
            numeric_fill_value
        )
    if categorical_columns:
        df_filled[categorical_columns] = df_filled[categorical_columns].fillna(
            categorical_fill_value
        )
    
    return df_filled

@task
def calculate_elephant_sentiment_score(
    df: AnyDataFrame,
    positive_columns: List[str],
    negative_columns: List[str]
) -> AnyDataFrame:
    positive_sentiment_mapping = {
        "I don't know": 0, 
        "Unspecified": 0,
        "Strongly disagree": 1,
        "Disagree": 2,
        "Neutral": 3,
        "Agree": 4,
        "Strongly agree": 5,
    }
    negative_sentiment_mapping = {
        "I don't know": 0, 
        "Unspecified": 0,
        "Strongly disagree": 5,
        "Disagree": 4,
        "Neutral": 3,
        "Agree": 2,
        "Strongly agree": 1,
    }
    
    df_scored = df.copy()
    for col in positive_columns:
        if col in df_scored.columns:
            score_col = f"{col}_score"
            df_scored[score_col] = df_scored[col].map(positive_sentiment_mapping)
        else:
            print(f"Warning: Column '{col}' not found in dataframe")
    
    for col in negative_columns:
        if col in df_scored.columns:
            score_col = f"{col}_score"
            df_scored[score_col] = df_scored[col].map(negative_sentiment_mapping)
        else:
            print(f"Warning: Column '{col}' not found in dataframe")
    
    # Get all score columns
    score_columns = [f"{col}_score" for col in positive_columns + negative_columns 
                     if col in df_scored.columns]
    
    df_scored['elephant_sentiment_score'] = df_scored[score_columns].mean(axis=1)
    df_scored['valid_response_count'] = df_scored[score_columns].notna().sum(axis=1)
    for col in score_columns:
        df_scored[col] = df_scored[col].astype('Int64')  
    
    df_scored['elephant_sentiment_score'] = df_scored['elephant_sentiment_score'].round(2)
    def map_overall_attitude(score):
        if pd.isna(score):
            return None
        elif score <= 1.5:
            return "Strongly disagree"
        elif score <= 2.5:
            return "Disagree"
        elif score <= 3.5:
            return "Neutral"
        elif score <= 4.5:
            return "Agree"
        else:
            return "Strongly agree"
    
    df_scored['overall_attitude'] = df_scored['elephant_sentiment_score'].apply(map_overall_attitude)
    df_scored = df_scored.sort_values(by="elephant_sentiment_score")
    return df_scored

@task
def merge_dataframes(
    left_df: AnyDataFrame,
    right_df: AnyDataFrame,
    on: str | list[str],
    how: Literal["left", "right", "inner", "outer"] = "left",
) -> AnyDataFrame:
    cols = [on] if isinstance(on, str) else on
    missing_left = [col for col in cols if col not in left_df.columns]
    missing_right = [col for col in cols if col not in right_df.columns]
    
    if missing_left:
        raise ValueError(f"Columns not found in left_df: {missing_left}")
    if missing_right:
        raise ValueError(f"Columns not found in right_df: {missing_right}")
    
    # Perform merge
    merged_df = pd.merge(left_df, right_df, on=on, how=how)
    return merged_df

@task
def perform_anova_analysis(
    dataframe: AnyDataFrame,
    target_column: str,
    factor_columns: List[str],
    anova_type: Literal[1, 2, 3] = 2,
) -> AnyDataFrame:
    if target_column not in dataframe.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataframe")
    
    missing_cols = [col for col in factor_columns if col not in dataframe.columns]
    if missing_cols:
        raise ValueError(f"Factor columns not found in dataframe: {missing_cols}")
    
    columns_to_check = [target_column] + factor_columns
    clean_df = dataframe[columns_to_check].dropna()
    
    if clean_df.empty:
        raise ValueError("No valid data after removing NaN values")

    factor_terms = [f"C({col})" for col in factor_columns]
    formula = f"{target_column} ~ {' + '.join(factor_terms)}"
    try:
        model = ols(formula, data=clean_df).fit()
    except Exception as e:
        raise ValueError(f"Error fitting OLS model: {str(e)}")
    anova_table = sm.stats.anova_lm(model, typ=anova_type)
    anova_df = anova_table.reset_index()
    anova_df.rename(columns={"index": "factor"}, inplace=True)
    anova_df["factor"] = anova_df["factor"].str.replace(r"C\((.*?)\)", r"\1", regex=True)
    numeric_columns = anova_df.select_dtypes(include=["float64"]).columns
    return anova_df

@task
def draw_boxplot(
    dataframe: DataFrame[JsonSerializableDataFrameModel],
    y_column: Annotated[
        str,
        Field(
            description="The name of the dataframe column to use for box plot values (y-axis)."
        ),
    ],
    x_column: Annotated[
        str | SkipJsonSchema[None],
        AdvancedField(
            default=None,
            description="The name of the dataframe column to group boxes by (x-axis categories). If None, creates a single box.",
        ),
    ] = None,
    color_column: Annotated[
        str | SkipJsonSchema[None],
        AdvancedField(
            default=None,
            description="The name of the dataframe column to color boxes with.",
        ),
    ] = None,
    orientation: Annotated[
        Literal["v", "h"] | SkipJsonSchema[None],
        AdvancedField(
            default="v",
            description="Orientation of the box plot. 'v' for vertical, 'h' for horizontal.",
        ),
    ] = "v",
    boxmode: Annotated[
        Literal["group", "overlay"] | SkipJsonSchema[None],
        AdvancedField(
            default="group",
            description="How to display boxes when there are multiple groups. 'group' places them side by side, 'overlay' overlaps them.",
        ),
    ] = "group",
    show_points: Annotated[
        Literal["all", "outliers", "suspectedoutliers", False] | SkipJsonSchema[None],
        AdvancedField(
            default=False,
            description="Whether to show data points. Options: 'all', 'outliers', 'suspectedoutliers', or False.",
        ),
    ] = False,
    plot_style: Annotated[
        PlotStyle | SkipJsonSchema[None],
        AdvancedField(
            default=None, description="Additional style kwargs passed to go.Box()."
        ),
    ] = None,
    layout_style: Annotated[
        LayoutStyle | SkipJsonSchema[None],
        AdvancedField(
            default=None,
            description="Additional kwargs passed to plotly.go.Figure(layout).",
        ),
    ] = None,
    widget_id: Annotated[
        str | SkipJsonSchema[None],
        Field(
            description="""\
            The id of the dashboard widget that this tile layer belongs to.
            If set this MUST match the widget title as defined downstream in create_widget tasks
            """,
            exclude=True,
        ),
    ] = None,
) -> Annotated[str, Field()]:
    import plotly.graph_objects as go

    layout_kwargs = layout_style.model_dump(exclude_none=True) if layout_style else {}
    style_kwargs = plot_style.model_dump(exclude_none=True) if plot_style else {}

    box_kwargs = {
        "y" if orientation == "v" else "x": dataframe[y_column],
        "boxpoints": show_points,
        "orientation": orientation,
        **style_kwargs,
    }
    if x_column:
        box_kwargs["x" if orientation == "v" else "y"] = dataframe[x_column]
        box_kwargs["name"] = x_column
    if color_column:
        box_kwargs["marker"] = {"color": dataframe[color_column]}

    if "marker_colors" in style_kwargs and x_column:
        traces = []
        unique_categories = dataframe[x_column].unique()
        marker_colors = style_kwargs.pop("marker_colors", [])
        
        for idx, category in enumerate(unique_categories):
            category_data = dataframe[dataframe[x_column] == category]
            color = marker_colors[idx % len(marker_colors)] if marker_colors else None
            
            trace_kwargs = {
                "y" if orientation == "v" else "x": category_data[y_column],
                "name": str(category),
                "boxpoints": show_points,
                "orientation": orientation,
                **{k: v for k, v in style_kwargs.items() if k != "marker_colors"},
            }
            
            if color:
                trace_kwargs["marker"] = {"color": color}
            
            traces.append(go.Box(**trace_kwargs))
        
        fig = go.Figure(data=traces)
    else:
        fig = go.Figure(data=[go.Box(**box_kwargs)])

    layout_kwargs["boxmode"] = boxmode
    fig.update_layout(**layout_kwargs)
    return fig.to_html(**ExportArgs(div_id=widget_id).model_dump(exclude_none=True))

@task
def draw_boxplot_and_persist(output_dir, df: AnyDataFrame, columns: Union[str, List[str]],y_column:str)->list:
    if df is None or df.empty:
        raise ValueError("DataFrame is None or empty")

    if isinstance(columns, str):
        columns = [columns]
    elif not isinstance(columns, list):
        raise ValueError("columns parameter must be a string or list of strings")
    
    missing_cols = [col for col in columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Columns not found in dataframe: {missing_cols}")
    
    bp_dir_list = []
    # Iterate through columns
    for column in columns:
        try:
            boxplot_chart = draw_boxplot(
                dataframe=df,
                y_column=y_column,
                x_column=column,
                show_points="outliers",
                plot_style=PlotStyle(
                    marker_colors=["#7eb0d5"]
                ),
                layout_style=LayoutStyle(
                    showlegend=False,
                    bargap=0.1,
                    font_size = 13,
                    font_color="#222222",
                ),
            )
            
            new_col = column.lower().replace(" ", "_")
            
            if boxplot_chart is None:
                print(f"An error occurred when generating `{column}` bar chart")
            else:
                file_path = persist_text( # persist_text
                    boxplot_chart, 
                    output_dir, 
                    f"{new_col}_boxplot_chart.html"
                )
                print(f"Successfully saved pie chart for `{column}` to {file_path}")
                bp_dir_list.append(file_path)
                
        except Exception as e:
            print(f"Error processing column `{column}`: {str(e)}")
            continue  
    return bp_dir_list



@task
def draw_scatter_chart(
    dataframe: DataFrame[JsonSerializableDataFrameModel],
    x_column: Annotated[
        str,
        Field(description="The name of the dataframe column for x-axis values."),
    ],
    y_column: Annotated[
        str,
        Field(description="The name of the dataframe column for y-axis values."),
    ],
    color_column: Annotated[
        str | SkipJsonSchema[None],
        AdvancedField(
            default=None,
            description="The name of the dataframe column to color points with.",
        ),
    ] = None,
    size_column: Annotated[
        str | SkipJsonSchema[None],
        AdvancedField(
            default=None,
            description="The name of the dataframe column to size points with.",
        ),
    ] = None,
    category_column: Annotated[
        str | SkipJsonSchema[None],
        AdvancedField(
            default=None,
            description="The name of the dataframe column to group points by (creates separate traces).",
        ),
    ] = None,
    scatter_style: Annotated[
        ScatterStyle | SkipJsonSchema[None],
        AdvancedField(
            default=None,
            description="Style configuration for scatter points.",
        ),
    ] = None,
    trendline_style: Annotated[
        TrendlineStyle | SkipJsonSchema[None],
        AdvancedField(
            default=None,
            description="Configuration for adding trendline to the scatter plot.",
        ),
    ] = None,
    layout_style: Annotated[
        LayoutStyle | SkipJsonSchema[None],
        AdvancedField(
            default=None,
            description="Additional kwargs passed to plotly.go.Figure(layout).",
        ),
    ] = None,
    widget_id: Annotated[
        str | SkipJsonSchema[None],
        Field(
            description="""\
            The id of the dashboard widget that this tile layer belongs to.
            If set this MUST match the widget title as defined downstream in create_widget tasks
            """,
            exclude=True,
        ),
    ] = None,
) -> Annotated[str, Field()]:
    import plotly.graph_objects as go
    import numpy as np

    layout_kwargs = layout_style.model_dump(exclude_none=True) if layout_style else {}
    scatter_style = scatter_style if scatter_style else ScatterStyle()
    trendline_style = trendline_style if trendline_style else TrendlineStyle()

    clean_df = dataframe[[x_column, y_column]].dropna()
    if clean_df.empty:
        raise ValueError(f"No valid data in columns {x_column} and {y_column}")

    marker_kwargs = {}
    if scatter_style.marker_color:
        marker_kwargs["color"] = scatter_style.marker_color
    if scatter_style.marker_size:
        marker_kwargs["size"] = scatter_style.marker_size
    if scatter_style.marker_symbol:
        marker_kwargs["symbol"] = scatter_style.marker_symbol
    if scatter_style.marker_opacity:
        marker_kwargs["opacity"] = scatter_style.marker_opacity

    if color_column:
        marker_kwargs["color"] = dataframe[color_column]
        marker_kwargs["colorscale"] = "Viridis"
        marker_kwargs["showscale"] = True

    if size_column:
        marker_kwargs["size"] = dataframe[size_column]
        marker_kwargs["sizemode"] = "diameter"
        marker_kwargs["sizeref"] = 2.0 * max(dataframe[size_column]) / (40.0**2)

    traces = []
    if category_column:
        categories = dataframe[category_column].unique()
        for category in categories:
            category_data = dataframe[dataframe[category_column] == category]
            scatter_trace = go.Scatter(
                x=category_data[x_column],
                y=category_data[y_column],
                mode=scatter_style.mode,
                name=str(category),
                marker=marker_kwargs,
            )
            traces.append(scatter_trace)
    else:
        scatter_trace = go.Scatter(
            x=dataframe[x_column],
            y=dataframe[y_column],
            mode=scatter_style.mode,
            name="Data Points",
            marker=marker_kwargs,
        )
        traces.append(scatter_trace)

    if trendline_style.enabled:
        x_vals = clean_df[x_column].values
        y_vals = clean_df[y_column].values

        if trendline_style.type == "ols":
            z = np.polyfit(x_vals, y_vals, 1)
            p = np.poly1d(z)
            x_trend = np.linspace(x_vals.min(), x_vals.max(), 100)
            y_trend = p(x_trend)
            trendline_name = f"OLS Trendline (y={z[0]:.3f}x+{z[1]:.3f})"

        elif trendline_style.type == "lowess":
            from scipy.signal import savgol_filter

            sorted_indices = np.argsort(x_vals)
            x_sorted = x_vals[sorted_indices]
            y_sorted = y_vals[sorted_indices]
            window_length = min(51, len(x_sorted) if len(x_sorted) % 2 == 1 else len(x_sorted) - 1)
            if window_length < 5:
                window_length = 5
            y_trend = savgol_filter(y_sorted, window_length, 3)
            x_trend = x_sorted
            trendline_name = "LOWESS Trendline"

        trendline_trace = go.Scatter(
            x=x_trend,
            y=y_trend,
            mode="lines",
            name=trendline_name,
            line=dict(
                color=trendline_style.color,
                width=trendline_style.width,
                dash=trendline_style.dash,
            ),
        )
        traces.append(trendline_trace)
    fig = go.Figure(data=traces)
    fig.update_layout(**layout_kwargs)
    return fig.to_html(**ExportArgs(div_id=widget_id).model_dump(exclude_none=True))

@task
def draw_ols_scatterplot_and_persist(
    output_dir, 
    df: AnyDataFrame, 
    columns: Union[str, List[str]],
    y_column:str
)->list:
    if df is None or df.empty:
        raise ValueError("DataFrame is None or empty")

    if isinstance(columns, str):
        columns = [columns]
    elif not isinstance(columns, list):
        raise ValueError("columns parameter must be a string or list of strings")
    
    missing_cols = [col for col in columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Columns not found in dataframe: {missing_cols}")
    
    ols_dir_list = []
    # Iterate through columns
    for column in columns:
        try:
            scatter_chart = draw_scatter_chart(
                dataframe=df,
                x_column=column,
                y_column=y_column,
                scatter_style=ScatterStyle(
                    marker_color="#6495ed",
                    marker_opacity=0.7,
                    marker_size=8,
                ),
                trendline_style=TrendlineStyle(
                    enabled=True,
                    type="ols",
                    color="red",
                ),
                layout_style=LayoutStyle(
                    showlegend=True,
                ),
            )
            
            # Clean column name for filename
            new_col = column.lower().replace(" ", "_")
            
            if scatter_chart is None:
                print(f"An error occurred when generating `{column}` bar chart")
            else:
                file_path = persist_text( # persist_text
                    scatter_chart, 
                    output_dir, 
                    f"{new_col}_ols_scatter_chart.html"
                )
                print(f"Successfully saved ols scatter chart for `{column}` to {file_path}")
                ols_dir_list.append(file_path)
                
        except Exception as e:
            print(f"Error processing column `{column}`: {str(e)}")
            continue  
    return ols_dir_list



@task
def draw_tukey_plot(
    dataframe: DataFrame[JsonSerializableDataFrameModel],
    value_column: Annotated[
        str,
        Field(
            description="The name of the dataframe column containing the continuous values to compare."
        ),
    ],
    group_column: Annotated[
        str,
        Field(
            description="The name of the dataframe column containing the group categories."
        ),
    ],
    tukey_style: Annotated[
        TukeyPlotStyle | SkipJsonSchema[None],
        AdvancedField(
            default=None,
            description="Style configuration for Tukey plot.",
        ),
    ] = None,
    layout_style: Annotated[
        LayoutStyle | SkipJsonSchema[None],
        AdvancedField(
            default=None,
            description="Additional kwargs passed to plotly.go.Figure(layout).",
        ),
    ] = None,
    widget_id: Annotated[
        str | SkipJsonSchema[None],
        Field(
            description="""\
            The id of the dashboard widget that this tile layer belongs to.
            If set this MUST match the widget title as defined downstream in create_widget tasks
            """,
            exclude=True,
        ),
    ] = None,
) -> Annotated[str, Field()]:
    import plotly.graph_objects as go
    from scipy import stats
    from itertools import combinations
    import numpy as np

    layout_kwargs = layout_style.model_dump(exclude_none=True) if layout_style else {}
    tukey_style = tukey_style if tukey_style else TukeyPlotStyle()
    clean_df = dataframe[[value_column, group_column]].dropna()
    if clean_df.empty:
        raise ValueError(
            f"No valid data in columns {value_column} and {group_column}"
        )
    groups = clean_df[group_column].unique()
    group_data = [
        clean_df[clean_df[group_column] == group][value_column].values
        for group in groups
    ]
    comparisons = []
    for group1, group2 in combinations(range(len(groups)), 2):
        data1 = group_data[group1]
        data2 = group_data[group2]

        mean1 = np.mean(data1)
        mean2 = np.mean(data2)
        mean_diff = mean1 - mean2

        n1, n2 = len(data1), len(data2)
        pooled_var = ((n1 - 1) * np.var(data1, ddof=1) + (n2 - 1) * np.var(data2, ddof=1)) / (
            n1 + n2 - 2
        )
        se = np.sqrt(pooled_var * (1 / n1 + 1 / n2))

        df = n1 + n2 - 2
        t_crit = stats.t.ppf((1 + tukey_style.confidence_level) / 2, df)
        ci_lower = mean_diff - t_crit * se
        ci_upper = mean_diff + t_crit * se
        is_significant = not (ci_lower <= 0 <= ci_upper)

        comparisons.append(
            {
                "group1": groups[group1],
                "group2": groups[group2],
                "mean_diff": mean_diff,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "is_significant": is_significant,
            }
        )

    comparisons.sort(key=lambda x: x["mean_diff"])
    comparison_labels = [
        f"{comp['group1']} - {comp['group2']}" for comp in comparisons
    ]
    traces = []
    for i, comp in enumerate(comparisons):
        color = (
            tukey_style.significant_color
            if comp["is_significant"]
            else tukey_style.non_significant_color
        )
        traces.append(
            go.Scatter(
                x=[comp["mean_diff"]],
                y=[i],
                mode="markers",
                marker=dict(
                    size=tukey_style.marker_size,
                    color=color,
                ),
                name=comparison_labels[i],
                showlegend=False,
                hovertemplate=(
                    f"<b>{comparison_labels[i]}</b><br>"
                    f"Mean Difference: {comp['mean_diff']:.3f}<br>"
                    f"95% CI: [{comp['ci_lower']:.3f}, {comp['ci_upper']:.3f}]<br>"
                    f"Significant: {'Yes' if comp['is_significant'] else 'No'}<br>"
                    "<extra></extra>"
                ),
            )
        )
        traces.append(
            go.Scatter(
                x=[comp["ci_lower"], comp["ci_upper"]],
                y=[i, i],
                mode="lines",
                line=dict(
                    color=color,
                    width=tukey_style.line_width,
                ),
                showlegend=False,
                hoverinfo="skip",
            )
        )
    traces.append(
        go.Scatter(
            x=[0, 0],
            y=[-0.5, len(comparisons) - 0.5],
            mode="lines",
            line=dict(color="gray", width=1, dash="dash"),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    traces.append(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker=dict(size=10, color=tukey_style.significant_color),
            name="Significant",
        )
    )
    traces.append(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker=dict(size=10, color=tukey_style.non_significant_color),
            name="Not Significant",
        )
    )

    fig = go.Figure(data=traces)
    default_layout = {
        "xaxis": {"title": "Mean Difference", "zeroline": True, "zerolinewidth": 2},
        "yaxis": {
            "title": "Group Comparisons",
            "tickmode": "array",
            "tickvals": list(range(len(comparisons))),
            "ticktext": comparison_labels,
        },
        "showlegend": True,
        "hovermode": "closest",
        "height": max(400, len(comparisons) * 40),
    }
    final_layout = {**default_layout, **layout_kwargs}
    fig.update_layout(**final_layout)
    return fig.to_html(**ExportArgs(div_id=widget_id).model_dump(exclude_none=True))

@task
def draw_tukey_plots_and_persist(
    output_dir, 
    df: AnyDataFrame, 
    columns: Union[str, List[str]],
    value_column:str
)->list:
    if df is None or df.empty:
        raise ValueError("DataFrame is None or empty")

    if isinstance(columns, str):
        columns = [columns]
    elif not isinstance(columns, list):
        raise ValueError("columns parameter must be a string or list of strings")
    
    missing_cols = [col for col in columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Columns not found in dataframe: {missing_cols}")
    
    tukey_dir_list = []
    # Iterate through columns
    for column in columns:
        try:
            tukey_chart = draw_tukey_plot(
                dataframe=df,
                value_column=value_column,
                group_column=column,
            )
            
            # Clean column name for filename
            new_col = column.lower().replace(" ", "_")
            
            if tukey_chart is None:
                print(f"An error occurred when generating `{column}` tukey chart")
            else:
                file_path = persist_text( #persist_text
                    tukey_chart, 
                    output_dir, 
                    f"{new_col}_tukey_chart.html"
                )
                print(f"Successfully saved tukey chart for `{column}` to {file_path}")
                tukey_dir_list.append(file_path)
                
        except Exception as e:
            print(f"Error processing column `{column}`: {str(e)}")
            continue  
    return tukey_dir_list

@task
def map_survey_columns(df: AnyDataFrame, cols: Union[str, List[str]]) -> AnyDataFrame:
    if isinstance(cols, str):
        cols = [cols]
    elif not isinstance(cols, list):
        raise ValueError("cols parameter must be a string or list of strings")
    
    df = map_survey_responses(
        df=df,
        columns=[
            "Respondent agreed to interview",
            "See more elephants now than before",
            "Do you change routes or schedules because of elephants",
            "Noticed signs of illness in wild animals",
            "Willingness to join future community dialogue",
            "Use measures to protect crops from elephants",
            "Ever been involved in or witnessed an elephant harmed",
            "Use measures to protect livestock from elephants",
            "Do you report elephant conflict incidents",
            "Pay into fence maintenance fund",
            "Benefit from the KCCDT Big Life electric fence",
            "Protect water sources from elephants",
        ],
        value_map={
            "yes": "Yes",
            "no": "No",
            "i_dont_know": "I don't know",
            "prefer_not_to_answer": "Prefer not to answer",
        }
    )
    
    df = map_values(
        df=df,
        column_name="Participant age",
        value_map={8.0: 17.0},
        missing_values="preserve"
    )
    
    df = map_values(
        df=df,
        column_name="Household size",
        value_map={-8.0: 8.0},
        missing_values="preserve"
    )

    # Map categorical variables
    df = map_values(
        df=df,
        column_name="Participant tribe",
        value_map={
            "luo": "Luo",
            "masai": "Maasai",
            "kikuya": "Kikuyu",
            "kamba": "Kamba",
            "prefer_not_to_answer": "Prefer not to answer",
            "other": "Other",
            "kisii": "Kisii",
            "somalis": "Somali",
            "luhya": "Luhya",
            "tanzanians": "Tanzanians"
        },
        missing_values="preserve"
    )

    df = map_values(
        df=df,
        column_name="Participant gender",
        value_map={
            "female": "Female",
            "male": "Male",
            "unknown": "Male"
        },
        missing_values="preserve"
    )

    df = map_values(
        df=df,
        column_name="Participant age group",
        value_map={
            "kidemi_mamas": "Kidemi Mamas",
            "Moran": "Moran",
            "senior_elder": "Senior Elder",
            "elder": "Elder",
            "junior_elder": "Junior Elder",
            "prefer_not_to_answer": "Prefer not to answer",
            "i_dont_know": "I don't know"
        },
        missing_values="preserve"
    )

    df = map_values(
        df=df,
        column_name="Years living in area",
        value_map={
            "1_10": "1–10 years",
            "11_20": "11–20 years",
            "21_30": "21–30 years",
            "31_40": "31–40 years",
            "41_50": "41–50 years",
            "50": "Over 50 years",
            "<1": "Less than 1 year",
            "prefer_not_to_answer": "Prefer not to answer"
        },
        missing_values="preserve"
    )

    df = map_values(
        df=df,
        column_name="Overall feelings about wildlife",
        value_map={
            "I_strongly_like_wildlife": "I strongly like wildlife",
            "I_like_wildlife": "I like wildlife",
            "I_am_neutral_toward_wildlife": "I am neutral toward wildlife",
            "I_dislike_wildlife": "I dislike wildlife",
            "I_strongly_dislike_wildlife": "I strongly dislike wildlife",
            "I_don't_know": "I don't know",
            "prefer_not_to_answer": "Prefer not to answer",
        },
        missing_values="preserve"
    )

    df = map_values(
        df=df,
        column_name="Opinion on having elephants in the area",
        value_map={
            "very_good": "Very good",
            "good": "Good",
            "neutral": "Neutral",
            "bad": "Bad",
            "very_bad": "Very bad",
            "prefer_not_to_answer": "Prefer not to answer",
            "i_dont_know": "I don't know"
        },
        missing_values="preserve"
    )

    df = map_values(
        df=df,
        column_name="How often seen elephants in the last year",
        value_map={
            "every_day": "Every day",
            "every_week": "Every week",
            "every_month": "Every month",
            "every_few months": "Every few months",
            "every_year": "Every year",
            "never": "Never",
            "i_dont_know": "I don't know",
        },
        missing_values="preserve"
    )

    df = map_values(
        df=df,
        column_name="What do you do when you encounter elephants on foot",
        value_map={
            "avoid_by_changing_route": "Avoid by changing route",
            "run_away": "Run away",
            "scare_it_away": "Scare it away",
            "other": "Other",
        },
        missing_values="preserve"
    )

    df = map_values(
        df=df,
        column_name="Reaction to hearing about elephants being harmed",
        value_map={
            "very_happy": "Very happy",
            "happy": "Happy",
            "neutral": "Neutral",
            "upset": "Upset",
            "very_upset": "Very upset",
            "i_dont_know": "I don't know",
            "prefer_not_to_answer": "Prefer not to answer",
        },
        missing_values="preserve"
    )

    df = map_values(
        df=df,
        column_name="Highest level of education",
        value_map={
            "primary": "Primary",
            "university": "University",
            "secondary": "Secondary",
            "none": "No formal education",
            "post_grad": "Postgraduate"
        },
        missing_values="preserve"
    )

    df = map_values(
        df=df,
        column_name="Greatest threat to your livestock",
        value_map={
            "drought": "Drought",
            "loss_from_wildlife": "Loss from wildlife",
            "disease": "Diseases",
            "not_applicable": "Other",
            "other": "Other"
        },
        missing_values="preserve"
    )

    df = map_values(
        df=df,
        column_name="Greatest threat to crop production",
        value_map={
            "drought": "Drought",
            "disease": "Crop disease",
            "damage_by_insects": "Damage by insects",
            "damage_by_wildlife": "Damage by wildlife",
            "soil_health": "Poor soil health",
            "labour_requirements": "Labour requirements",
            "other": "Other",
        },
        missing_values="preserve"
    )

    df = map_values(
        df=df,
        column_name="Which intervention would help you in future",
        value_map={
            "protection_for_water_pipes_tanks_or_boreholes": "Protection for water pipes, tanks, or boreholes",
            "elephant_awareness_training": "Elephant awareness training",
            "elephant_deterrent": "Elephant deterrent methods",
            "adjustment_to_walking_patterns_for_daily_activities": "Adjust walking patterns for daily activities",
            "adjustment_to_grazing_pattern": "Adjust grazing patterns",
        },
        missing_values="preserve"
    )

    df = map_values(
        df=df,
        column_name="Marital status",
        value_map={
            # --- Married (monogamous or generic) ---
            'Married': 'Married',
            'Married ': 'Married',
            'Married.': 'Married',
            ' married ': 'Married',
            'Maried': 'Married',
            'Marriage': 'Married',
            'Marriage ': 'Married',
            'Married (eamishe)': 'Married',
            'Married ( eamishe)': 'Married',
            'Married ( eamishe ': 'Married',
            'Married (eamishe )': 'Married',
            'Married eamishe': 'Married',
            'Married eamishe ': 'Married',
            'Eamishe ': 'Married',
            'aiema(married)': 'Married',
            'Iama': 'Married',
            'Iama ': 'Married',
            'Iams': 'Married',
            'iamishe': 'Married',
            'iamishe ': 'Married',
            'Iamishiee': 'Married',
            'Iamishe': 'Married',
            'Iamishe ': 'Married',
            '1amishe': 'Married',
            'yes': 'Married',
            'Yes': 'Married',
            'Yes ': 'Married',
            'Eeeh': 'Married',
            'Married eamaki': 'Married',
            'Married (eamaki)': 'Married',
            'Married (eamaki) ': 'Married',
            'Married monogamous ': 'Married',
            'married (eamishe)': 'Married',
            'Married ( eamishe) ': 'Married',
            'married ': 'Married',
            'Married (eamishe) ': 'Married',
            
            # --- Married Polygamous ---
            'Married polygamous ': 'Married Polygamous',
            'Married poligamy ': 'Married Polygamous',
            'Married polygamously': 'Married Polygamous',
            'Polygamous ': 'Married Polygamous',
            'Poligamy ': 'Married Polygamous',
            'Marriage poligamy ': 'Married Polygamous',
            'Married polygamously ': 'Married Polygamous',

            # --- Separated / Divorced ---
            'Separated': 'Separated/Divorced',
            'Separated ': 'Separated/Divorced',
            'Divorced': 'Separated/Divorced',
            'Divorced ': 'Separated/Divorced',
            'Married but separated ': 'Separated/Divorced',

            # --- Widowed ---
            'Widow': 'Widowed',
            'Widow ': 'Widowed',
            'Widowed': 'Widowed',
            'Widowed ': 'Widowed',
            'Widower ': 'Widowed',

            # --- Single / Not Married ---
            'Single': 'Single',
            'Single ': 'Single',
            'single': 'Single',
            'single ': 'Single',
            'single(itu)': 'Single',
            'Single (itu)': 'Single',
            'Single ( itu aemisho)': 'Single',
            'Single mother': 'Single',
            'Single mother.': 'Single',
            'Single mother ': 'Single',
            'Single parent': 'Single',
            'Single parent ': 'Single',
            'Not married': 'Single',
            'Not Married': 'Single',
            'Not married ': 'Single',
            'Notmarried ': 'Single',
            ' Not Married': 'Single',
            'Not Married ': 'Single',
            'No married ': 'Single',
            'Not. Married ': 'Single',
            'Not married as': 'Single',
            'No': 'Single',
            'Not ': 'Single',
            'No( itu)': 'Single',
            'None ': 'Single',
            'Itu': 'Single',
            'iti': 'Single',
            'Iti': 'Single',
            'Left wife and kids': 'Single',
            
            # --- Unknown / Invalid ---
            '57': 'Unspecified',
            'iama': 'Unspecified'
        },
        missing_values="preserve"
    )

    df = map_values(
        df=df,
        column_name="Land tenure",
        value_map={
            "own": "Own",
            "prefer_not_to_say": "Prefer not to answer",
            "seasonal_lease": "Seasonal lease",
            "annual_lease": "Annual lease",
            "profit_share": "Profit share",
            "employed_labour": "Employed labour",
            "not_applicable": "Not applicable",
            "other": "Other",
        },
        missing_values="preserve"
    )
    
    return df

@task
def exclude_geom_outliers(
    df: AnyDataFrame,
    z_threshold: float = 3.0,
) -> AnyDataFrame:
    
    if df.empty:
        print("Warning: Input dataframe is empty")
        return df
    
    if len(df) < 4:
        print(f"Warning: Too few points ({len(df)}) for reliable outlier detection. Returning original data.")
        return df
    
    if "geometry" not in df.columns:
        raise ValueError("DataFrame must have a 'geometry' column")
    df_work = df.copy()

    df_work["x"] = df_work.geometry.x
    df_work["y"] = df_work.geometry.y

    centroid_x = df_work["x"].mean()
    centroid_y = df_work["y"].mean()

    df_work["dist_from_center"] = np.sqrt(
        (df_work["x"] - centroid_x)**2 + (df_work["y"] - centroid_y)**2
    )
    
    dist_mean = df_work["dist_from_center"].mean()
    dist_std = df_work["dist_from_center"].std()
    
    if dist_std == 0:
        print("Warning: All points at same location (std=0). No outliers removed.")
        return df.copy()
    
    z_scores = (df_work["dist_from_center"] - dist_mean) / dist_std
    mask = np.abs(z_scores) < z_threshold
    df_clean = df[mask].copy()
    outliers_count = (~mask).sum()
    print(f"  - Total points: {len(df)}")
    print(f"  - Outliers removed: {outliers_count} ({outliers_count/len(df)*100:.1f}%)")
    print(f"  - Points retained: {len(df_clean)}")
    return df_clean


@task
def exclude_value(df:AnyDataFrame,column:str , value:Union[str, int, float]) -> AnyDataFrame:
    df_filtered = df[df[column] != value].copy()
    return cast(AnyDataFrame, df_filtered)
