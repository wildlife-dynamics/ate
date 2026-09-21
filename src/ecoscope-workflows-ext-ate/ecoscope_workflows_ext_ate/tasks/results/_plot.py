import warnings
import pandas as pd
from wt_registry import register
from pydantic import Field, BaseModel
from typing_extensions import Literal
from typing import List, Dict, Annotated
from pydantic.json_schema import SkipJsonSchema
from ecoscope.platform.tasks.results._ecoplot import ExportArgs,LayoutStyle
from ecoscope.platform.annotations import (
    AdvancedField,
    AnyDataFrame,
    DataFrame,
    JsonSerializableDataFrameModel,
)

warnings.filterwarnings("ignore")


class TukeyPlotStyle(BaseModel):
    significant_color: Annotated[str | SkipJsonSchema[None], AdvancedField(default="#ff6b6b")] = "#ff6b6b"
    non_significant_color: Annotated[str | SkipJsonSchema[None], AdvancedField(default="#4ecdc4")] = "#4ecdc4"
    marker_size: Annotated[int | SkipJsonSchema[None], AdvancedField(default=8)] = 8
    line_width: Annotated[int | SkipJsonSchema[None], AdvancedField(default=2)] = 2


class ScatterStyle(BaseModel):
    marker_size: Annotated[int | SkipJsonSchema[None], AdvancedField(default=None)] = None
    marker_color: Annotated[str | SkipJsonSchema[None], AdvancedField(default=None)] = None
    marker_symbol: Annotated[str | SkipJsonSchema[None], AdvancedField(default=None)] = None
    marker_opacity: Annotated[float | SkipJsonSchema[None], AdvancedField(ge=0.0, le=1.0, default=None)] = None
    mode: Annotated[str | SkipJsonSchema[None], AdvancedField(default="markers")] = "markers"


class TrendlineStyle(BaseModel):
    enabled: Annotated[bool, AdvancedField(default=False)] = False
    type: Annotated[
        Literal["ols", "lowess"] | SkipJsonSchema[None],
        AdvancedField(
            default="ols",
            description="Type of trendline. 'ols' for linear regression, 'lowess' for locally weighted  smoothing.",
        ),
    ] = "ols"
    color: Annotated[str | SkipJsonSchema[None], AdvancedField(default="red")] = "red"
    width: Annotated[int | SkipJsonSchema[None], AdvancedField(default=2)] = 2
    dash: Annotated[str | SkipJsonSchema[None], AdvancedField(default="solid")] = "solid"


class LikertStyle(BaseModel):
    colors: Annotated[
        Dict[str, str] | SkipJsonSchema[None],
        AdvancedField(
            default=None,
            description="Dictionary mapping response categories to colors. If None, uses default color scheme.",
        ),
    ] = None
    neutral_categories: Annotated[
        List[str] | SkipJsonSchema[None],
        AdvancedField(
            default=None,
            description="Categories considered neutral (displayed on positive side). "
            "If None, auto-detects or uses middle category.",
        ),
    ] = None
    show_percentages: Annotated[
        bool | SkipJsonSchema[None],
        AdvancedField(default=True, description="Whether to show percentage labels on bars."),
    ] = True
    sort_questions: Annotated[
        bool | SkipJsonSchema[None],
        AdvancedField(default=False, description="Whether to sort questions by response distribution."),
    ] = False
    sort_by: Annotated[
        Literal["positive", "negative", "name"] | SkipJsonSchema[None],
        AdvancedField(
            default="positive",
            description="How to sort questions when sort_questions is enabled.",
        ),
    ] = "positive"


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
    df: AnyDataFrame, positive_responses: List[str], negative_responses: List[str], sort_by: str
) -> AnyDataFrame:
    if sort_by == "name":
        return df[sorted(df.columns)]
    scores = {}
    for col in df.columns:
        counts = df[col].value_counts()
        total = counts.sum()

        if sort_by == "positive":
            score = sum(counts.get(r, 0) for r in positive_responses) / total if total > 0 else 0
        elif sort_by == "negative":
            score = sum(counts.get(r, 0) for r in negative_responses) / total if total > 0 else 0
        else:
            score = 0

        scores[col] = score
    sorted_cols = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    return df[sorted_cols]


def _merge_layout(default_layout: dict, layout_kwargs: dict) -> dict:
    """Merge a caller-supplied layout on top of chart defaults.

    A plain ``{**default_layout, **layout_kwargs}`` would let a partial
    ``xaxis``/``yaxis`` override (e.g. just a custom title) silently wipe out
    the rest of that axis's defaults (tickmode/tickvals/ticktext, etc.), since
    dict unpacking replaces the whole nested value rather than merging into it.
    """
    merged = {**default_layout, **layout_kwargs}
    for axis in ("xaxis", "yaxis"):
        if isinstance(default_layout.get(axis), dict) and isinstance(layout_kwargs.get(axis), dict):
            merged[axis] = {**default_layout[axis], **layout_kwargs[axis]}
    return merged


@register()
def draw_likert_chart(
    dataframe: DataFrame[JsonSerializableDataFrameModel],
    title: Annotated[
        str | SkipJsonSchema[None],
        AdvancedField(default=None, description="Chart title."),
    ] = None,
    response_order: Annotated[
        List[str] | SkipJsonSchema[None],
        AdvancedField(
            default=None,
            description="Order of responses from most negative to most positive. If None, inferred from data.",
        ),
    ] = None,
    likert_style: Annotated[
        LikertStyle | SkipJsonSchema[None],
        AdvancedField(default=None, description="Style configuration for the Likert chart."),
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

    if dataframe is None or dataframe.empty:
        raise ValueError("DataFrame is None or empty")

    likert_style = likert_style if likert_style else LikertStyle()
    layout_kwargs = layout_style.model_dump(exclude_none=True) if layout_style else {}

    response_order = response_order if response_order is not None else _infer_response_order(dataframe)
    colors = likert_style.colors if likert_style.colors is not None else _generate_color_scheme(response_order)
    neutral_categories = likert_style.neutral_categories

    mid_point = len(response_order) // 2

    if neutral_categories is None:
        if len(response_order) % 2 == 1:
            neutral_categories = [response_order[mid_point]]
            negative_responses = response_order[:mid_point]
            positive_responses = response_order[mid_point + 1 :]
        else:
            neutral_categories = []
            negative_responses = response_order[:mid_point]
            positive_responses = response_order[mid_point:]
    else:
        negative_responses = [
            r for r in response_order if r not in neutral_categories and response_order.index(r) < mid_point
        ]
        positive_responses = [
            r for r in response_order if r not in neutral_categories and response_order.index(r) >= mid_point
        ]

    if likert_style.sort_questions:
        dataframe = _sort_dataframe(dataframe, positive_responses, negative_responses, likert_style.sort_by)

    show_percentages = likert_style.show_percentages
    percentages_df = dataframe.apply(lambda col: col.value_counts() / len(col) * 100, axis=0)
    fig = go.Figure()
    legend_rank = len(response_order)
    for response in reversed(negative_responses):
        values = []
        for col in dataframe.columns:
            percentage = percentages_df.loc[response, col] if response in percentages_df.index else 0
            values.append(-percentage)

        fig.add_trace(
            go.Bar(
                x=values,
                y=dataframe.columns,
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
        for col in dataframe.columns:
            percentage = percentages_df.loc[response, col] if response in percentages_df.index else 0
            values.append(percentage)

        fig.add_trace(
            go.Bar(
                x=values,
                y=dataframe.columns,
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

    min_height = 400
    height_per_question = 60
    chart_height = max(min_height, len(dataframe.columns) * height_per_question)

    default_layout = {
        "title": dict(
            text=title,
            font=dict(size=24, color="black", family="Arial"),
            x=0.5,
            xanchor="center",
            y=0.98,
            yanchor="top",
        ),
        "barmode": "relative",
        "height": chart_height,
        "width": 1200,
        "yaxis_autorange": "reversed",
        "bargap": 0.15,
        "margin": dict(l=300, r=100, t=150, b=80),
        "xaxis": dict(
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
        "yaxis": dict(
            tickfont=dict(size=12, color="black", family="Arial"),
            linecolor="black",
            showline=True,
            gridcolor="#e2e8f0",
        ),
        "showlegend": True,
        "legend": dict(
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
        "plot_bgcolor": "white",
        "paper_bgcolor": "white",
    }
    final_layout = _merge_layout(default_layout, layout_kwargs)
    fig.update_layout(**final_layout)

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

    return fig.to_html(**ExportArgs(div_id=widget_id).model_dump(exclude_none=True))

@register()
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

    base_marker_kwargs = {}
    if scatter_style.marker_color:
        base_marker_kwargs["color"] = scatter_style.marker_color
    if scatter_style.marker_size:
        base_marker_kwargs["size"] = scatter_style.marker_size
    if scatter_style.marker_symbol:
        base_marker_kwargs["symbol"] = scatter_style.marker_symbol
    if scatter_style.marker_opacity:
        base_marker_kwargs["opacity"] = scatter_style.marker_opacity

    if color_column:
        base_marker_kwargs["colorscale"] = "Viridis"
        base_marker_kwargs["showscale"] = True

    if size_column:
        base_marker_kwargs["sizemode"] = "diameter"
        base_marker_kwargs["sizeref"] = 2.0 * max(dataframe[size_column]) / (40.0**2)

    def marker_kwargs_for(data: AnyDataFrame) -> dict:
        marker_kwargs = dict(base_marker_kwargs)
        if color_column:
            marker_kwargs["color"] = data[color_column]
        if size_column:
            marker_kwargs["size"] = data[size_column]
        return marker_kwargs

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
                marker=marker_kwargs_for(category_data),
            )
            traces.append(scatter_trace)
    else:
        scatter_trace = go.Scatter(
            x=dataframe[x_column],
            y=dataframe[y_column],
            mode=scatter_style.mode,
            name="Data Points",
            marker=marker_kwargs_for(dataframe),
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

    default_layout = {
        "xaxis": {"title": x_column},
        "yaxis": {"title": y_column},
        "showlegend": bool(category_column) or trendline_style.enabled,
        "hovermode": "closest",
        "plot_bgcolor": "white",
        "paper_bgcolor": "white",
    }
    final_layout = _merge_layout(default_layout, layout_kwargs)
    fig.update_layout(**final_layout)
    return fig.to_html(**ExportArgs(div_id=widget_id).model_dump(exclude_none=True))



@register()
def draw_tukey_chart(
    comparisons: Annotated[
        AnyDataFrame,
        Field(
            description="Pairwise group comparisons, as returned by compute_tukey_comparisons: "
            "group1, group2, mean_diff, ci_lower, ci_upper, p_value, is_significant."
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

    if comparisons is None or comparisons.empty:
        raise ValueError("comparisons DataFrame is None or empty")

    tukey_style = tukey_style if tukey_style else TukeyPlotStyle()
    layout_kwargs = layout_style.model_dump(exclude_none=True) if layout_style else {}

    records = comparisons.to_dict("records")
    comparison_labels = [f"{r['group1']} - {r['group2']}" for r in records]
    traces = []
    for i, r in enumerate(records):
        color = tukey_style.significant_color if r["is_significant"] else tukey_style.non_significant_color
        traces.append(
            go.Scatter(
                x=[r["mean_diff"]],
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
                    f"Mean Difference: {r['mean_diff']:.3f}<br>"
                    f"95% CI: [{r['ci_lower']:.3f}, {r['ci_upper']:.3f}]<br>"
                    f"Adjusted p-value: {r['p_value']:.3f}<br>"
                    f"Significant: {'Yes' if r['is_significant'] else 'No'}<br>"
                    "<extra></extra>"
                ),
            )
        )
        traces.append(
            go.Scatter(
                x=[r["ci_lower"], r["ci_upper"]],
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
            y=[-0.5, len(records) - 0.5],
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
            "tickvals": list(range(len(records))),
            "ticktext": comparison_labels,
        },
        "showlegend": True,
        "hovermode": "closest",
        "height": max(400, len(records) * 40),
    }
    final_layout = _merge_layout(default_layout, layout_kwargs)
    fig.update_layout(**final_layout)
    return fig.to_html(**ExportArgs(div_id=widget_id).model_dump(exclude_none=True))