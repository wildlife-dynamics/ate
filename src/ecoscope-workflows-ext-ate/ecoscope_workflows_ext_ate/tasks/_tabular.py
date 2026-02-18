import warnings
import numpy as np
import pandas as pd
from typing_extensions import Literal
from ecoscope_workflows_core.decorators import task
from ecoscope_workflows_core.annotations import AnyDataFrame
from typing import Iterable, Optional, Union,List,Dict,Annotated,cast
from ecoscope_workflows_core.tasks.transformation._mapping import map_values

from pydantic import Field
from pydantic.json_schema import SkipJsonSchema

warnings.filterwarnings("ignore")

def _normalize_columns(
        df: AnyDataFrame, 
        columns: Optional[Union[str, Iterable[str]]]
        ):
    if columns is None:
        # default: all object dtype cols
        return list(df.select_dtypes(include="object").columns)
    if isinstance(columns, str):
        return [columns]
    return list(columns)

@task
def convert_object_to_value(
    df: AnyDataFrame, 
    columns: Optional[Union[str, Iterable[str]]] = None
    ) -> AnyDataFrame:
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
def convert_object_to_string(
    df: AnyDataFrame, 
    columns: Optional[Union[str, Iterable[str]]] = None
    ) -> AnyDataFrame:
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
        "I dont know": 0, 
        "Unspecified": 0,
        "Strongly disagree": 1,
        "Disagree": 2,
        "Neutral": 3,
        "Agree": 4,
        "Strongly agree": 5,
    }
    negative_sentiment_mapping = {
        "I dont know": 0, 
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
            "i_dont_know": "I dont know",
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
            "i_dont_know": "I dont know"
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
            "I_don't_know": "I dont know",
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
            "i_dont_know": "I dont know"
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
            "i_dont_know": "I dont know",
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
            "i_dont_know": "I dont know",
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
def exclude_value(df:AnyDataFrame,column:str , value:Union[str, int, float]) -> AnyDataFrame:
    df_filtered = df[df[column] != value].copy()
    return cast(AnyDataFrame, df_filtered)