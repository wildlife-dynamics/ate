from pydantic import Field
from wt_registry import register
from typing import Annotated, cast
from ecoscope.platform.annotations import  AnyDataFrame

@register()
def round_column_values(
    df: AnyDataFrame,
    column: Annotated[str, Field(description="The column to round")],
    output_column: Annotated[str, Field(description="The output column name")],
    decimals: Annotated[int, Field(description="Number of decimal places to round to")] = 0,
) -> AnyDataFrame:
    df[output_column] = df[column].round(decimals)
    return cast(AnyDataFrame, df)