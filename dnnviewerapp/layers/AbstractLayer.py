import string

from ..SimpleColorScale import SimpleColorScale
from ..widgets import tabs

import numpy as np
import plotly.graph_objects as go
import dash_core_components as dcc
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

    def set_xoffset(self, xoffset: float):
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

    # @abstract
    def plot(self, fig: go.Figure):
        # to override
        return

    def plot_topn_connections(self, previous_layer, topn, active_units, backward):
        """ Plot layers' top n connections"""
        # to override
        # return the list of strongest and the list of shapes of connectors
        return [], []

    # @abstract
    def get_layer_title(self):
        """ Get layer description to be included in the Dash Column """
        return []

    # @abstract
    def get_layer_tabs(self, previous_active: string):
        """ Get the layer tab bar and layout function """
        return tabs.make('bottom-layer', {}, previous_active,
                         # The graph needs always to be defined at init to check associated callback
                         html.Div(dcc.Graph(id='bottom-layer-figure'), hidden=True))

    # @abstract
    def get_layer_tab_content(self, active_tab):
        """ Get the content of the selected tab """
        return html.Div()

    # @abstract
    def get_unit_title(self, unit_idx: int):
        """ Get unit description to be included in the Dash Column """
        return [html.H5(('Unit #%s' % unit_idx) +
                        (' (%s)' % self.unit_names[unit_idx] if self.unit_names is not None else ""))]

    # @abstract
    def get_unit_tabs(self, unit_idx: int, previous_active: string):
        """ Get the unit tab bar and layout function """
        return [*tabs.make('bottom-unit', {}, previous_active),
                # The graph needs always to be defined at init to check associated callback
                html.Div(dcc.Graph(id='bottom-unit-figure'), hidden=True)]

    # @abstract
    def get_unit_tab_content(self, unit_idx: int, active_tab: string):
        """ Get the content of the selected tab """
        return html.Div()

    def get_unit_description(self, unit_idx: int):
        """ Get layer Unit description to be included in a Dash Column """
        return [html.H5(('Unit #%s' % unit_idx) +
                        (' (%s)' % self.unit_names[unit_idx] if self.unit_names is not None else ""))]

    def _get_y_offset(self):
        """ index of the first unit (lowest y) """
        return -self.num_unit * self.spacing_y / 2
