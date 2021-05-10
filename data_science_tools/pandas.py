"""Pandas related."""
import pandas as pd


def value_counts(s: pd.Series) -> pd.Series:
    """Shortcut for Series.value_counts()."""
    return s.value_counts(dropna=False, sort=False)


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


def cat_with_codes(s: pd.Series, mapping: dict, ordered=False):
    """Create a categorical with codes/descriptions."""
    mapping = mapping.copy()

    dtype = pd.CategoricalDtype(list(mapping.keys()), ordered)
    series_as_dtype = s.astype(dtype)

    return series_as_dtype.cat.rename_categories(mapping)
