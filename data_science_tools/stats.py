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

__DEBUG__ = False


def debug(*args, **kwargs):
    if __DEBUG__:
        print(*args, **kwargs)


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


class CMHResult(object):
    """Represents the result of a Cochran-Mantel-Haenszel Chi2 analysis."""

    def __init__(self, STATISTIC, df, p, var1, var2, stratifier, alpha=0.05):
        """
        Initialize a new CMHResult.

            STATISTIC: X2 statistic
            df: degrees of freedom
            p: p-value
        """
        self.STATISTIC = STATISTIC
        self.df = df
        self.p = p
        self.var1 = var1
        self.var2 = var2
        self.stratifier = stratifier
        self.alpha = alpha

    def __repr__(self):
        stat = round(self.STATISTIC, 5)
        pval = round(self.p, 4)
        df = self.df

        return textwrap.dedent(f"""
                Cochran-Mantel-Haenszel Chi2 test

        "{self.var1}" x "{self.var2}", stratified by "{self.stratifier}"

        Cochran-Mantel-Haenszel M^2 = {stat}, df = {df}, p-value = {pval}
        """)

    def _repr_html_(self):
        stat = round(self.STATISTIC, 5)
        pval = round(self.p, 4)
        df = self.df
        tpl = f"""
        <div style="font-family: courier; font-size: 10pt; padding: 0px 10px;">
            <div style="text-align:center">
                Cochran-Mantel-Haenszel Chi&#x00B2; test
            </div>

            <div>
                <b>{self.var1}</b> x <b>{self.var2}</b>,
                stratified by <b>{self.stratifier}</b>
            </div>
            <div>
                Cochran-Mantel-Haenszel
                    M^2 = {stat},
                    df = {df},
                    p-value = <b>{pval}</b>
            </div>
        </div>
        """

        if pval > self.alpha:
            return box(tpl, '#efefef', '#cfcfcf')
        return box(tpl, '#b0cbe9', '#4393e1')


def CMH(df: pd.DataFrame, var: str, outcome: str, stratifier: str, raw=False):
    """Compute the CMH statistic.

    Based on "Categorical Data Analysis", page 295 by Agresti (2002) and
    R implementation of mantelhaen.test().
    """
    df = df.copy()
    df[outcome] = df[outcome].astype('category')
    df[var] = df[var].astype('category')
    df[stratifier] = df[stratifier].astype('category')

    # Compute contingency table size KxIxJ
    I = len(df[outcome].cat.categories)
    J = len(df[var].cat.categories)
    K = len(df[stratifier].cat.categories)

    contingency_tables = np.zeros((I, J, K), dtype='float')

    # Create stratified contingency tables
    for k in range(K):
        cat = df[stratifier].cat.categories[k]

        subset = df.loc[df[stratifier] == cat, [var, outcome]]
        xs = pd.crosstab(subset[outcome], subset[var], dropna=False)
        contingency_tables[:, :, k] = xs

    # Compute the actual CMH
    STATISTIC, df, pval = CMH_numpy(contingency_tables)

    if raw:
        return STATISTIC, df, pval

    return CMHResult(STATISTIC, df, pval, var, outcome, stratifier)


def CMH_numpy(X):
    """Compute the CMH statistic.

    Based on "Categorical Data Analysis", page 295 by Agresti (2002) and
    R implementation of mantelhaen.test().
    """
    # I: nr. of rows
    # J: nr. of columns
    # K: nr. of strata
    # ⚠️ Note: this does *not* match the format used when printing!

    I, J, K = X.shape

    debug(f"I: {I}, J: {J}, K: {K}")
    debug()

    df = (I - 1) * (J - 1)
    debug(f'{df} degree(s) of freedom')

    # Initialize m and n to a vector(0) of length df
    n = np.zeros(df)
    m = np.zeros(df)
    V = np.zeros((df, df))

    # iterate over the strata
    for k in range(K):
        debug(f'partial {k}')
        # f holds partial contigency table k
        f = X[:, :, k]

        # debuggin'
        debug('  f:')
        debug(f)
        debug()

        # Sum of *all* values in the partial table
        ntot = f.sum()
        debug(f'  ntot: {ntot}')

        # Compute the sum over all row/column entries *excluding* the last
        # entry. The last entries are excluded, as they hold redundant
        # information in combination with the row/column totals.
        colsums = f.sum(axis=0)[:-1]
        rowsums = f.sum(axis=1)[:-1]

        debug('  rowsums:', rowsums)
        debug('  colsums:', colsums)

        # f[-I, -J] holds the partial matrix, excluding the last row & column.
        # The result is reshaped into a vector.
        debug('  f[:-1, :-1].reshape(-1): ', f[:-1, :-1].reshape(-1))
        n = n + f[:-1, :-1].reshape(-1)

        # Take the outer product of the row- and colsums, divide it by the
        # total of the partial table. Yields a vector of length df. This holds
        # the expected value under the assumption of conditional independence.
        m_k = (np.outer(rowsums, colsums) / ntot).reshape(-1)
        m = m + m_k
        debug('  m_k:', m_k)
        debug()

        # V_k holds the null covariance matrix (matrices).
        k1 = np.diag(ntot * colsums)[:J, :J] - np.outer(colsums, colsums)
        k2 = np.diag(ntot * rowsums)[:I, :I] - np.outer(rowsums, rowsums)

        debug('np.kron(k1, k2):')
        debug(np.kron(k1, k2))
        debug()

        V_k = np.kron(k1, k2) / (ntot**2 * (ntot - 1))

        debug('  V_k:')
        debug(V_k)

        V = V + V_k

        debug()

    # Subtract the mean from the entries
    n = n - m
    debug(f'n: {n}')
    debug()

    debug('np.linalg.solve(V, n):')
    debug(np.linalg.solve(V, n))
    debug()

    STATISTIC = np.inner(n, np.linalg.solve(V, n).transpose())
    debug('STATISTIC:', STATISTIC)

    pval = 1 - stats.chi2.cdf(STATISTIC, df)

    return STATISTIC, df, pval


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
