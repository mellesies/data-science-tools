"""
Plotting stuff!
"""
from typing import Union

import pandas as pd
import altair as alt


def hist(df: Union[pd.DataFrame, pd.Series], varname: str = None):
    """Plot a histogram."""
    if isinstance(df, pd.Series):
        assert df.index.name != '', "Series should have a name!"

        varname = df.name
        df = pd.DataFrame(df)

        # return df

    return alt.Chart(df).mark_bar().encode(
        alt.X(varname),
        alt.Y('count()'),
        tooltip='count()',
    )
