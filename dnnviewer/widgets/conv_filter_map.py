from ..theming.Theme import Theme
from ..theming.SimpleColorScale import SimpleColorScale

from plotly.subplots import make_subplots
import plotly.graph_objects as go

import numpy as np


def figure(values, theme: Theme, color_scale: SimpleColorScale):
    """ Show a map of convolutional filters """
    num_maps = min(values.shape[2], 12)
    # Number of columns on map depending on the kernel size
    if values.shape[1] < 4:
        num_cols = 3
    elif values.shape[1] < 6:
        num_cols = 2
    else:
        num_cols = 1
    num_rows = max(num_cols, int(np.ceil(num_maps / num_cols)))

    titles = [str(i) for i in range(num_maps)]

    fig = make_subplots(rows=num_rows, cols=num_cols, subplot_titles=titles,
                        shared_xaxes=True, shared_yaxes=True,
                        horizontal_spacing=0.02, vertical_spacing=0.06)

    # Draw filters as subplots
    for i in range(num_maps):
        fig.add_trace(go.Heatmap(z=values[:, :, i], coloraxis="coloraxis"),
                      row=(i // num_cols) + 1, col=(i % num_cols) + 1)

    title = 'Filters' + (' (%d out of %d)' % (num_maps, values.shape[2]) if values.shape[2] > num_maps else '')
    fig.update_layout(margin=theme.bottom_figure_margins,
                      title=dict(text=title, font=dict(size=14)),
                      coloraxis=color_scale.as_dict(),
                      template=theme.plotly,
                      font=dict(size=12))
    return fig
