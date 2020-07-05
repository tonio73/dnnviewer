from ..theming.Theme import Theme
from ..theming.SimpleColorScale import SimpleColorScale

import numpy as np
import plotly.graph_objects as go


def figure(values, num_unit, unit_names, theme: Theme, color_scale: SimpleColorScale, selected=None, title='Unit'):
    """ Create a bar chart with min and max over layer """

    value_min = np.amin(values, axis=0)
    value_max = np.amax(values, axis=0)
    value_mean = np.mean(np.absolute(values), axis=0)
    hover_text_min = ['Min of unit %d' % idx for idx in np.arange(num_unit)] if unit_names is None else unit_names
    hover_text_max = ['Max of unit %d' % idx for idx in np.arange(num_unit)] if unit_names is None else unit_names
    hover_text_ma = ['Mean absolute of unit %d' % idx for idx in np.arange(num_unit)] if unit_names is None \
        else unit_names
    if num_unit < 64:
        # For small layers, display as bar traces
        min_trace = go.Bar(x=value_min, hovertext=hover_text_min, hoverinfo='text',
                           marker=color_scale.as_dict(value_min), customdata=['layer'], name='Min')
        max_trace = go.Bar(x=value_max, hovertext=hover_text_max, hoverinfo='text',
                           marker=color_scale.as_dict(value_max), customdata=['layer'], name='Max')
    else:
        # For large layers, display as line traces
        min_trace = go.Scatter(x=value_min, hovertext=hover_text_min, hoverinfo='text',
                               mode='lines', marker=color_scale.as_dict(value_min), customdata=['layer'], name='Min')
        max_trace = go.Scatter(x=value_max, hovertext=hover_text_max, hoverinfo='text',
                               mode='lines', marker=color_scale.as_dict(value_max), customdata=['layer'], name='Max')

    ma_trace = go.Scatter(x=value_mean, hovertext=hover_text_ma, hoverinfo='text',
                          mode='lines', customdata=['layer'], name='Mean')

    fig = go.Figure(data=[min_trace, max_trace, ma_trace])
    if selected is not None and selected < num_unit:
        annotation_title = ("#%d (%s)" % (selected, unit_names[selected])) if unit_names else ("#%d" % selected)
        annotations = [dict(x=0, y=selected,
                            xref="x", yref="y",
                            text=annotation_title,
                            showarrow=True, arrowhead=3, arrowcolor="#aaa",
                            bgcolor="#666", borderpad=4,
                            ax=-30, ay=-20)]
    else:
        annotations = []

    fig.update_layout(margin=theme.bottom_figure_margins,
                      title=dict(text=title + ' min, max, mean absolute', font=dict(size=14)),
                      yaxis_title_text='Layer unit',
                      bargap=0,  # gap between bars of adjacent location coordinates)
                      showlegend=False,
                      template=theme.plotly,
                      annotations=annotations)
    return fig
