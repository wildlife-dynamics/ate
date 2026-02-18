import warnings
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from typing_extensions import Literal
from ecoscope.base.utils import hex_to_rgba
from ecoscope_workflows_core.decorators import task
from ecoscope_workflows_core.tasks.io import persist_text
from ecoscope_workflows_core.annotations import AnyDataFrame
from typing import Optional, Union,List,Dict,Annotated
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
        df = _sort_dataframe(df,positive_responses, 
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
    
    default_color = '#008b8b'  # Dark cyan
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
                    font_size=13,
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
                bar_dir_list.append(file_path)
                
        except Exception as e:
            print(f"Error processing column `{column}`: {str(e)}")
            continue  
    return bar_dir_list

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
            description="The name of the dataframe column to group boxes by (x-axis categories).",
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
            description="How to display boxes when there are multiple groups.",
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
        box_kwargs["marker"] = {"color": dataframe[color_column].iloc[0]}

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
def draw_boxplot_and_persist(
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
                tukey_dir_list.append(file_path)
                
        except Exception as e:
            print(f"Error processing column `{column}`: {str(e)}")
            continue  
    return tukey_dir_list