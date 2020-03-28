from ..SimpleColorScale import SimpleColorScale

import numpy as np
import plotly.graph_objects as go
import dash_html_components as html


class AbstractLayer:
    """ Abstract layer representation in Viewer """

    def __init__(self, name, num_unit=0, weights=None,
                 plotly_theme='plotly_dark', link_color_scale=SimpleColorScale(),
                 unit_names=None):
        self.name = name
        self.num_unit = num_unit
        self.weights = weights
        self.unit_names = unit_names
        self.link_color_scale = link_color_scale
        self.plotly_theme = plotly_theme
        self.spacing_y = 1.
        self.xoffset = 0
        return

    def set_xoffset(self, xoffset):
        self.xoffset = xoffset

    def get_unit_position(self, unit_idx, at_output=False):
        """ Get single or vector of unit positions """
        if isinstance(unit_idx, int):
            x = self.xoffset
        else:
            x = self.xoffset * np.ones(len(unit_idx))

        return x, self._get_y_offset() + self.spacing_y * unit_idx

    def get_positions(self):
        """ Get all unit positions """
        return self.get_unit_position(np.arange(self.num_unit))

    def plot(self, fig: go.Figure):
        # to override
        return

    def plot_topn_connections(self, previous_layer, topn, active_units, backward):
        """ Plot layers' top n connections"""
        # to override
        # return the list of strongest and the list of shapes of connectors
        return [], []

    # @abstract
    def get_layer_description(self):
        """ Get layer description to be included in the Dash Column """
        return []

    # @abstract
    def get_layer_figure(self, mode):
        """ Figure illustrating the layer """
        fig = go.Figure()
        fig.update_layout(margin=dict(l=10, r=10, b=30, t=40),
                          showlegend=False,
                          template=self.plotly_theme)
        return fig

    def get_unit_description(self, unit_idx: int):
        """ Get layer Unit description to be included in a Dash Column """
        return [html.H5(('Unit #%s' % unit_idx) +
                        (' (%s)' % self.unit_names[unit_idx] if self.unit_names is not None else ""))]

    def _get_y_offset(self):
        return -self.num_unit * self.spacing_y / 2

    def _plot_title(self, fig, y_offset=-10):
        fig.add_trace(go.Scatter(x=[self.xoffset], y=[self._get_y_offset() + y_offset], text=[self.name],
                                 textposition="bottom center", mode="text"))
