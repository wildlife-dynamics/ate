import numpy as np
import pandas as pd
from wt_registry import register
from ecoscope.platform.annotations import AnyDataFrame


@register()
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
            _, bin_edges = pd.qcut(clean_data, q=n_bins, retbins=True, duplicates="drop")
            bin_edges[0] = 0  # Start from 0
            bin_edges[-1] = float("inf")  # Extend to infinity

            # Create labels
            labels = []
            for i in range(len(bin_edges) - 1):
                if bin_edges[i + 1] == float("inf"):
                    labels.append(f"{int(bin_edges[i])}+")
                else:
                    labels.append(f"{int(bin_edges[i])}-{int(bin_edges[i+1])}")

            return bin_edges.tolist(), labels
        except Exception as e:
            print(f"{e}")
            bin_edges = np.linspace(0, max_val * 1.1, n_bins + 1)
            bin_edges[-1] = float("inf")
            labels = [
                f"{int(bin_edges[i])}-{int(bin_edges[i+1]) if bin_edges[i+1] != float('inf') else '+'}"
                for i in range(len(bin_edges) - 1)
            ]
            return bin_edges.tolist(), labels

    rows = []
    total = len(df)

    for col in columns_of_interest:
        if col not in df.columns:
            continue

        numeric = pd.to_numeric(df[col], errors="coerce")
        if numeric.notna().sum() > 0 and numeric.notna().sum() / len(numeric) > 0.5:
            bins, labels = create_bins(numeric)

            if bins and labels:
                binned = pd.cut(numeric, bins=bins, labels=labels)
                counts = binned.value_counts().reindex(labels, fill_value=0)

                for cat, cnt in counts.items():
                    pct = (cnt / total) * 100 if total else 0
                    rows.append(
                        {
                            "Demographic Variable": col,
                            "Categories": str(cat),
                            "Number of responses": f"{pct:.2f}% (n={int(cnt)})",
                        }
                    )

                stats = numeric.dropna()
                if not stats.empty:
                    stats_text = (
                        f"(mean={stats.mean():.1f}; median={stats.median():.0f}; "
                        f"SD={stats.std():.2f}; max={stats.max():.0f}; min={stats.min():.0f})"
                    )
                else:
                    stats_text = "(no numeric data)"

                rows.append({"Demographic Variable": "", "Categories": "", "Number of responses": stats_text})
        else:
            series = df[col].fillna("No Response").astype(str)
            counts = series.value_counts(dropna=False)

            for cat, cnt in counts.items():
                pct = (cnt / total) * 100 if total else 0
                rows.append(
                    {
                        "Demographic Variable": col,
                        "Categories": str(cat),
                        "Number of responses": f"{pct:.2f}% (n={int(cnt)})",
                    }
                )

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
