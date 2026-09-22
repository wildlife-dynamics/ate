import pandas as pd
from pydantic import Field
from typing import Annotated
from wt_registry import register
from pydantic.json_schema import SkipJsonSchema
from ecoscope.platform.annotations import (
    AdvancedField, 
    AnyDataFrame,
    DataFrame,
    JsonSerializableDataFrameModel
)

@register()
def compute_tukey_comparisons(
    dataframe: DataFrame[JsonSerializableDataFrameModel],
    value_column: Annotated[
        str,
        Field(description="The name of the dataframe column containing the continuous values to compare."),
    ],
    group_column: Annotated[
        str,
        Field(description="The name of the dataframe column containing the group categories."),
    ],
    confidence_level: Annotated[
        Annotated[float, Field(ge=0.0, le=1.0)] | SkipJsonSchema[None],
        AdvancedField(
            default=0.95,
            description="Confidence level for the pairwise comparison intervals.",
        ),
    ] = 0.95,
) -> Annotated[
    AnyDataFrame,
    Field(
        description="Pairwise group comparisons (Tukey HSD) with columns: "
        "group1, group2, mean_diff, ci_lower, ci_upper, p_value, is_significant."
    ),
]:
    from statsmodels.stats.multicomp import pairwise_tukeyhsd

    if value_column == group_column:
        raise ValueError("value_column and group_column must be different columns")

    if not pd.api.types.is_numeric_dtype(dataframe[value_column]):
        raise ValueError(
            f"value_column '{value_column}' must be numeric, got dtype {dataframe[value_column].dtype}"
        )

    clean_df = dataframe[[value_column, group_column]].dropna()
    if clean_df.empty:
        raise ValueError(f"No valid data in columns {value_column} and {group_column}")

    groups = clean_df[group_column].unique()
    if len(groups) < 2:
        raise ValueError(f"Need at least 2 groups in '{group_column}' to compare, found {len(groups)}")

    result = pairwise_tukeyhsd(
        endog=clean_df[value_column].to_numpy(),
        groups=clean_df[group_column].to_numpy(),
        alpha=1 - confidence_level,
    )
    table = result.summary().data
    comparisons = pd.DataFrame(table[1:], columns=table[0]).rename(
        columns={
            "meandiff": "mean_diff",
            "p-adj": "p_value",
            "lower": "ci_lower",
            "upper": "ci_upper",
            "reject": "is_significant",
        }
    )
    comparisons["is_significant"] = comparisons["is_significant"].astype(bool)
    columns = ["group1", "group2", "mean_diff", "ci_lower", "ci_upper", "p_value", "is_significant"]
    return comparisons[columns].sort_values("mean_diff").reset_index(drop=True)
