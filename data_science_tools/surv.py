from typing import List

import numpy as np
import pandas as pd


def discretize_age(df: pd.DataFrame, colname='leeft',
                   min=0, max=120, step=10) -> pd.Series:
    bins = list(range(min, max, step))
    labels = []

    for idx, b in enumerate(bins):
        labels.append(f"[{b+1}, {b+10}]")

    # We need 1 fewer labels than edges
    labels = labels[:-1]

    return pd.cut(df[colname], bins=bins, labels=labels, right=True)


def discretize_survival(df: pd.DataFrame,
                        cutoffs: List[int] = None,
                        event_col='vitstat',
                        time_col='vitfup',
                        multiplier=365,
                        cutoff_unit='y',
                        prefix='surv_',
                        inplace=False,
                        ) -> pd.DataFrame:
    """
    Args:
        df: DataFrame
        cutoffs: cutoffs, measured in time_col unit * multiplier
        time_col: duration of follow up in *days*
        event_col: vital status at last follow up (0: alive, 1: deceased)

    Returns:
        pd.DataFrame
    """

    # For each cutoff, we'll try to determine who lived/died.
    # - for cases with vitstat == 1 (deceased at last follow up), we can fill
    #   out all cutoffs
    # - cases with vitstat == 0 (alive), we can only fill out those where
    #   cutoff < vitfup
    if cutoffs is None:
        cutoffs = [1, 2, 3, 4, 5]

    # vitstat == 1: deceased at last follow up
    # vitstat == 0: alive at last follow up
    idx_alive = df[event_col] == 0
    idx_deceased = df[event_col] == 1

    if inplace is False:
        df = df.copy()

    labels = []

    for cutoff in cutoffs:
        label = f'{prefix}{cutoff:02}{cutoff_unit}'
        labels.append(label)

        cutoff_in_units = cutoff * multiplier

        df[label] = np.nan

        # in deceased cases, we can fill out *all* the cutoffs
        df.loc[idx_deceased, label] = df[time_col] > cutoff_in_units

        # For patients still alive, we can only say something up until the
        # last moment of follow up
        idx_alive_at_cutoff = idx_alive & (df[time_col] > cutoff_in_units)
        df.loc[idx_alive_at_cutoff, label] = True

    if inplace is False:
        return df[labels]


# Alias ...
discretize_follow_up = discretize_survival
