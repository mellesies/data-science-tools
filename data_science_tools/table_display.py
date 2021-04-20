"""Stuff related to showing tables."""
from IPython.display import display

from lifelines import CoxPHFitter


def add_significance_meter(df, pcol='p', colname='sign.',
                           thresholds=[0.05, 0.005, 0.0005],
                           p_precision=4):
    """Add a column indicating the level of significance."""
    # We don't want to mess up the original dataframe.
    df = df.copy()

    # Sort thresholds from large --> small
    thresholds.sort(reverse=True)

    # Initialize all entries to ''
    df[colname] = ''

    # Add the meter
    for idx, th in enumerate(thresholds):
        df.loc[df[pcol] <= th, colname] = '*' * (idx + 1)

    if p_precision:
        p_tpl = f"{{:.{p_precision}f}}"
        df[pcol] = df[pcol].map(p_tpl.format)
        df[pcol] = df[pcol].replace('nan', '')

    # Done ...
    return df


def table(df, caption='', table_nr=0, precision=2):
    """Display table with more style then a panda."""
    if not hasattr(table, "table_nr"):
        table.table_nr = 0

    table.table_nr += 1
    global_precision = None

    df = df.copy()

    if isinstance(precision, dict):
        for colname, col_precision in precision.items():
            if colname == '__all__':
                global_precision = col_precision
            else:
                fmt = '{' + f':.{col_precision}f' + '}'
                df[colname] = df[colname].map(fmt.format)
                df[colname] = df[colname].replace('nan', '')

    styled = df.style.set_properties(**{
        'font-family': 'monospace',
        # 'border': '1px solid red',
    })

    table_nr = table_nr or table.table_nr

    if caption and table_nr:
        caption = f'<b>Table {table_nr}:</b> {caption}'
    elif caption:
        caption = f'<b>Table:</b> {caption}'

    styled = styled.set_caption(caption)

    if global_precision is not None:
        styled = styled.set_precision(global_precision)
    elif isinstance(precision, int):
        styled = styled.set_precision(precision)

    styled = styled.format(None, na_rep="")
    # styled = styled.hide_index()

    styled = styled.set_table_styles([
        {
            'selector': '',
            'props': [
                ('border', '1px solid #cfcfcf'),
                ('margin', '5px'),
            ]
        }, {
            'selector': 'th',
            'props': [
                ('font-family', 'monospace'),
                ('font-weight', 'bold'),
                ('vertical-align', 'top'),
            ]
        }, {
            'selector': 'caption',
            'props': [
                ('caption-side', 'bottom'),
                ('font-family', 'monospace'),
                ('text-align', 'left'),
                ('color', '#666666'),
                ('padding', '10px 20px'),
            ]
        },
    ])

    display(styled)


def display_cph_summary(cph_or_summary, references=None, caption='',
                        table_nr=''):
    """Display a pretty formatted version of the CPH summary."""
    if isinstance(cph_or_summary, CoxPHFitter):
        summary = cph_or_summary.summary
    else:
        summary = cph_or_summary

    # Just to be sure
    summary = summary.copy()
    del summary['z']
    del summary['-log2(p)']
    del summary['coef lower 95%']
    del summary['coef upper 95%']

    if references:
        summary = add_references_to_cox_summary(summary, references)

    # add the column 'sign.' that indicates the level of significance
    summary = add_significance_meter(summary)

    summary.index.name = 'variable'
    summary = summary.reset_index()

    # summary.style.set_caption('This is a caption')
    styled = summary.style.apply(highlight, axis=1)

    styled = styled.set_properties(**{
        'font-family': 'monospace',
        #'border': '1px solid red',
    })

    if caption and table_nr:
        caption = f'<b>Table {table_nr}:</b> {caption}'
    elif caption:
        caption = f'<b>Table:</b> {caption}'

    styled = styled.set_caption(caption)
    styled = styled.set_precision(3)

    styled = styled.format(None, na_rep="")
    styled = styled.hide_index()
    styled = styled.set_table_styles([
        {
            'selector': '',
            'props': [
                ('border', '1px solid #cfcfcf'),
            ]
        }, {
            'selector': 'th',
            'props': [
                ('font-family', 'monospace'),
                ('font-weight', 'bold'),
                ('vertical-align', 'top'),
            ]
        }, {
            'selector': 'caption',
            'props': [
                ('caption-side', 'bottom'),
                ('font-family', 'monospace'),
                ('text-align', 'left'),
                ('color', '#666666'),
                ('padding', '10px 40px'),
            ]
        },
    ])

    display(styled)
