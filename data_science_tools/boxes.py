"""Jupyter Notebook Tools - functions for displaying div's and such."""
from IPython.display import display, HTML


def box(text, background_color, border_color, raw=False):
    """Draw text in a box.

    Arguments:
        background_color: CSS-compatible background color.
        border_color: CSS-compatible border color.
        raw: return HTML as a string iff True; will call display(...)
             iff False.
    """
    html_ = f"""
            <div style="
                background-color: {background_color};
                border: 1px solid {border_color};
                padding: 5px;"
            >
                {text}
            </div>
        """

    if raw is False:
        return display(HTML(html_))

    return html_


def create_box_functions():
    """Create functions to draw text in a box with predifined colors."""
    box_colors = {
        'info': ["#B0CBE9", '#4393E1'],
        'warn': ["#FEC958", 'orange'],
        'error': ["#ffaaaa", 'red'],
    }

    functions = {}

    for type_ in box_colors:

        def f(background_color, border_color):
            def func(text, background_color=background_color,
                     border_color=border_color, **kwargs):
                return box(text, background_color, border_color, **kwargs)

            return func

        background_color = box_colors[type_][0]
        border_color = box_colors[type_][1]

        functions[type_] = f(background_color, border_color)

    return functions


# Create the functions info(), warn() and error()
for name, func in create_box_functions().items():
    locals()[name] = func

del name, func

# info('info')
# warn('warn')
# error('error')
