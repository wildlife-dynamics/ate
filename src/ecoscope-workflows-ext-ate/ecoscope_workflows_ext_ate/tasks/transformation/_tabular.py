from pydantic import Field
from functools import reduce
from wt_registry import register
from typing import Annotated, cast, Literal
from ecoscope.platform.annotations import AnyDataFrame
from operator import add, floordiv, mod, mul, pow, sub, truediv

# Genuine binary operators — folded left-to-right with reduce
binary_operations = {
    "add": add,
    "subtract": sub,
    "multiply": mul,
    "divide": truediv,
    "floor_divide": floordiv,
    "modulo": mod,
    "power": pow,
}

# Row-wise aggregations across all selected columns
aggregate_operations = {
    "min": "min",
    "max": "max",
    "mean": "mean",
}

Operations = Literal[
    "add",
    "subtract",
    "multiply",
    "divide",
    "floor_divide",
    "modulo",
    "power",
    "min",
    "max",
    "mean",
]


@register()
def round_column_values(
    df: AnyDataFrame,
    column: Annotated[str, Field(description="The column to round")],
    output_column: Annotated[str, Field(description="The output column name")],
    decimals: Annotated[int, Field(description="Number of decimal places to round to")] = 0,
) -> AnyDataFrame:
    df[output_column] = df[column].round(decimals)
    return cast(AnyDataFrame, df)


@register()
def apply_arithmetic_operation_over_columns(
    df: AnyDataFrame,
    columns: Annotated[list[str], Field(description="Column names to combine, left to right", min_length=2)],
    output_column: Annotated[str, Field(description="The output column name")],
    operation: Annotated[Operations, Field(description="The arithmetic operation to apply")],
) -> AnyDataFrame:
    if operation in aggregate_operations:
        method = aggregate_operations[operation]
        df[output_column] = getattr(df[columns], method)(axis=1)
    else:
        op = binary_operations[operation]
        df[output_column] = reduce(op, (df[c] for c in columns))
    return cast(AnyDataFrame, df)
