"""Pandas related."""
import pandas as pd


def remove_unused_categories(df: pd.DataFrame, inplace=False) -> pd.DataFrame:
    """Remove unused categories from all (categorical) columns in a DataFrame.
    """
    if inplace is False:
        df = df.copy()

    for col in df.columns:
        try:
            df[col].cat.remove_unused_categories(inplace=True)
        except Exception:
            pass

    if inplace is False:
        return df
