import numpy as np
import plotly.graph_objects as go


def figure(values, num_unit, unit_names, plotly_theme, title='Unit'):
    """ Create a bar chart with min and max over layer """

    value_min = np.amin(values, axis=0)
    value_max = np.amax(values, axis=0)
    value_mean = np.mean(np.absolute(values), axis=0)
    hover_text_min = ['Min of unit %d' % idx for idx in np.arange(num_unit)] if unit_names is None else unit_names
    hover_text_max = ['Max of unit %d' % idx for idx in np.arange(num_unit)] if unit_names is None else unit_names
    hover_text_ma = ['Mean absolute of unit %d' % idx for idx in np.arange(num_unit)] if unit_names is None else unit_names
    fig = go.Figure(data=[go.Bar(x=value_min, hovertext=hover_text_min, hoverinfo='text',
                                 customdata=['layer'], name='Min'),
                          go.Bar(x=value_max, hovertext=hover_text_max, hoverinfo='text',
                                 customdata=['layer'], name='Max'),
                          go.Scatter(x=value_mean, hovertext=hover_text_ma, hoverinfo='text',
                                     mode='lines', customdata=['layer'], name='Mean')
                          ])
    fig.update_layout(margin=dict(l=10, r=10, b=30, t=40),  # noqa: E741
                      title_text= title + ' min, max, mean absolute',
                      yaxis_title_text='Layer unit',
                      bargap=0,  # gap between bars of adjacent location coordinates)
                      showlegend=False,
                      template=plotly_theme)
    return fig
