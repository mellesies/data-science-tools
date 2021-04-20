"""
Stats stuff!
"""
import textwrap

import numpy as np
import pandas as pd
from scipy import stats

from IPython.display import display

from .boxes import *
from .table_display import *

class Chi2Result(object):

    def __init__(self, name1: str, name2: str, xs: pd.DataFrame, dof: int,
                 p: float, alpha=0.05):
        """Create a new Chi2Result instance."""
        self.name1 = name1
        self.name2 = name2
        self.xs = xs
        self.dof = dof
        self.p = p
        self.alpha = alpha

    def __repr__(self):
        """Return a string representation of this result."""
        if self.p <= self.alpha:
            p_conclusion = f'p ≤ {self.alpha}'
        else:
            p_conclusion = f'p > {self.alpha}'

        s = f"""
        Chi2 analysis between {self.name1} and {self.name2}
            p = {self.p:.4f} with {self.dof} degree(s) of freedom.
            {p_conclusion}
        """
        return textwrap.dedent(s)

    def _repr_html_(self):
        """Return an HTML representation of this result."""
        if self.p <= self.alpha:
            p_conclusion = f'p ≤ {self.alpha}'
        else:
            p_conclusion = f'p > {self.alpha}'

        tpl = f"""
            <div style="font-family: courier; padding: 0px 10px;">
                <div style="text-align:center">
                    Chi&#x00B2; analysis between <b>{self.name1}</b> and
                    <b>{self.name2}</b></div>
                <div>p-value: <b>{self.p:.4f}</b> with
                    <b>{self.dof}</b> degree(s) of freedom.</div>
                <div>{p_conclusion}</div>
            </div>
        """
        if self.p <= self.alpha:
            return info(tpl, raw=True)

        return box(tpl, '#efefef', '#cfcfcf', raw=True)


def Chi2(col1: pd.Series, col2: pd.Series, show_crosstab=False) -> Chi2Result:
    """Compute the Chi2 statistic."""
    xs = pd.crosstab(col1, col2)
    _, p, dof, expected = stats.chi2_contingency(xs)

    if show_crosstab:
        display(xs)

    return Chi2Result(col1.name, col2.name, xs, dof, p)


def table1(df, vars, outcome, p_name='p', p_precision=None, title=''):
    """Prepare Table 1"""
    def replace(string, dict_):
        for key, replacement in dict_.items():
            if string == key:
                return replacement

        return string

    # We're going to create multiple tables, one for each variable.
    tables = []

    col2 = df[outcome]

    totals = col2.value_counts()
    headers = {
        header: f'{header} (n={total})' for header, total in totals.iteritems()
    }

    # Iterate over the variables
    for v in vars:
        col1 = df[v]

        # Crosstab with absolute numbers
        x1 = pd.crosstab(col1, col2)

        # Crosstab with percentages
        x2 = pd.crosstab(col1, col2, normalize='columns')
        x2 = (x2 * 100).round(1)

        # Chi2 is calculated using absolute nrs.
        chi2, p, dof, expected = stats.chi2_contingency(x1)

        # Combine absolute nrs. with percentages in a single cell.
        xs = x1.astype('str') + ' (' + x2.applymap('{:3.1f}'.format) + ')'

        # Add the totals ('n={total}') to the headers
        xs.columns = [replace(h, headers) for h in list(xs.columns)]

        # If title is provided, we'll add a level to the column index and put
        # it there (on top).
        if title:
            colidx = pd.MultiIndex.from_product(
                [[title, ], list(xs.columns)],
            )

            xs.columns = colidx

        # Add the p-value in a new column, but only in the top row.
        xs[p_name] = ''

        if p_precision:
            p_tpl = f"{{:.{p_precision}f}}"
            xs.iloc[0, len(xs.columns) - 1] = p_tpl.format(p)
        else:
            xs[p_name] = np.nan
            xs.iloc[0, len(xs.columns) - 1] = p

        # Prepend the name of the current variable to the row index, so we can
        # concat the tables later ...
        xs.index = pd.MultiIndex.from_product(
            [[v, ], list(xs.index)],
            names=['variable', 'values']
        )

        tables.append(xs)

    return pd.concat(tables)
