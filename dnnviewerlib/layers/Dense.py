from .AbstractLayer import AbstractLayer
from ..Connector import Connector
from ..Statistics import Statistics
from ..SimpleColorScale import SimpleColorScale
from ..widgets import layer_minimax_graph

import numpy as np
import plotly.graph_objects as go
import dash_core_components as dcc
import dash_html_components as html


class Dense(AbstractLayer):
    """ Dense (aka fully connected) layer of n units """
    """ Assume 2D weight tensor with dimensions: previous layer unit, self unit """

    def __init__(self, name, num_unit, weights, plotly_theme, link_color_scale=SimpleColorScale(), unit_names=None):
        assert weights.ndim == 2
        assert num_unit == weights.shape[1]

        AbstractLayer.__init__(self, name, num_unit, weights, plotly_theme, link_color_scale, unit_names)

    # @override
    def plot(self, fig):
        x, y = self.get_positions()
        hover_text = ['%d' % idx for idx in np.arange(self.num_unit)] if self.unit_names is None else self.unit_names
        fig.add_trace(go.Scatter(x=x, y=y, hovertext=hover_text, mode='markers', hoverinfo='text', name=self.name))

    # @override
    def plot_topn_connections(self, backward_layer, topn, active_units, backward):
        if self.weights is None:
            return np.empty(0), []

        # KO if flatten output on backward_layer
        # assert backward_layer.num_unit == self.weights.shape[0]

        if backward:
            strongest_idx, strongest = Statistics.get_strongest(self.weights[:, active_units],
                                                                min(topn, backward_layer.num_unit))
            # For each of the top n, create a vector of connectors and plot it
            to_indexes = np.tile(active_units, strongest.shape[0])

            strongest_idx = strongest_idx.ravel()
            strongest = strongest.ravel()

            connectors = Connector(backward_layer, self,
                                   strongest_idx, to_indexes, strongest,
                                   self.link_color_scale)
        else:
            strongest_idx, strongest = Statistics.get_strongest(self.weights.T[:, active_units],  # <--
                                                                min(topn, backward_layer.num_unit))

            # For each of the top n, create a vector of connectors and plot it
            from_indexes = np.tile(active_units, strongest.shape[0])

            strongest_idx = strongest_idx.ravel()
            strongest = strongest.ravel()

            connectors = Connector(backward_layer, self,
                                   from_indexes, strongest_idx, strongest,
                                   self.link_color_scale)

        return np.unique(strongest_idx), connectors.get_shapes()

    # @override
    def get_layer_description(self):
        return [html.H5("Dense '%s'" % self.name),
                html.Ul([html.Li("%d units" % self.num_unit)])]

    # @override
    def get_layer_figure(self, mode):
        if mode == 'weights':
            return layer_minimax_graph.graph(self.weights, self.num_unit, self.unit_names, self.plotly_theme)
        return AbstractLayer.get_layer_figure(self, mode)

    # @override
    def get_unit_description(self, unit_idx):
        w = self.weights[:, unit_idx]
        fig = go.Figure(data=[go.Histogram(x=w)])
        fig.update_layout(margin=dict(l=10, r=10, b=30, t=40),
                          title_text='Weight histogram',
                          xaxis_title_text='Amplitude',
                          # yaxis_title_text='Count',
                          bargap=0.2,  # gap between bars of adjacent location coordinates)
                          template=self.plotly_theme)

        return AbstractLayer.get_unit_description(self, unit_idx) + \
               [html.Ul([html.Li("%d coefficients" % len(w))]),
                dcc.Graph(id='unit-%s-%d-histogram' % (self.name, unit_idx), figure=fig)]
