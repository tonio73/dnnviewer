import numpy as np
import plotly.graph_objects as go


def figure(weights, num_unit, unit_names, plotly_theme):
    """ Create a bar chart with min and max over layer """

    w_min = np.amin(weights, axis=0)
    w_max = np.amax(weights, axis=0)
    hover_text_min = ['Min unit %d' % idx for idx in np.arange(num_unit)] if unit_names is None else unit_names
    hover_text_max = ['Max unit %d' % idx for idx in np.arange(num_unit)] if unit_names is None else unit_names
    fig = go.Figure(data=[go.Bar(x=w_min, hovertext=hover_text_min, hoverinfo='text',
                                 customdata=['layer'], name='Min'),
                          go.Bar(x=w_max, hovertext=hover_text_max, hoverinfo='text',
                                 customdata=['layer'], name='Max')])
    fig.update_layout(margin=dict(l=10, r=10, b=30, t=40),  # noqa: E741
                      title_text='Unit min-max weights',
                      yaxis_title_text='Layer unit',
                      # yaxis_title_text='Count',
                      bargap=0,  # gap between bars of adjacent location coordinates)
                      showlegend=False,
                      template=plotly_theme)
    return fig
