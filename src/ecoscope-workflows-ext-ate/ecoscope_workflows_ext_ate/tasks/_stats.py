
import warnings
import statsmodels.api as sm
from typing_extensions import Literal
from statsmodels.formula.api import ols
from ecoscope_workflows_core.decorators import task
from ecoscope_workflows_core.annotations import AnyDataFrame
from typing import List

warnings.filterwarnings("ignore")

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
    return anova_df