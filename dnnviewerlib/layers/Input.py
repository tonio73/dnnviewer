from .AbstractLayer import AbstractLayer

import plotly.graph_objects as go
import dash_html_components as html

import numpy as np


class Input(AbstractLayer):
    """ Simple placeholder layer to act as input to the network """
    def __init__(self, name, num_unit, plotly_theme, unit_names=None):
        AbstractLayer.__init__(self, name, num_unit, None, plotly_theme, unit_names=unit_names)

    # @override
    def plot(self, fig):
        x, y = self.get_positions()
        hover_text = ['%d' % idx for idx in np.arange(self.num_unit)] if self.unit_names is None else self.unit_names
        fig.add_trace(go.Scatter(x=x, y=y, hovertext=hover_text, mode='markers', hoverinfo='text', name=self.name))

    # @override
    def get_layer_description(self):
        return [html.H5("Input '%s'" % self.name),
                html.Ul([html.Li("%d units" % self.num_unit)])
                ]
