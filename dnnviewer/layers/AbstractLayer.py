from ..widgets import tabs
from ..theming.Theme import Theme

import numpy as np
import plotly.graph_objects as go
import dash_core_components as dcc
import dash_html_components as html


class AbstractLayer:
    """ Abstract layer representation in Viewer """

    def __init__(self, name, num_unit=0, weights=None, grads=None,
                 theme: Theme = Theme(),
                 unit_names=None):
        self.name = name
        self.num_unit = num_unit
        self.weights = weights
        self.grads = grads
        self.unit_names = unit_names
        self.theme = theme
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
    def get_layer_tabs(self, previous_active: str = None):
        """ Get the layer tab bar and layout function """
        return tabs.make('bottom-layer', {}, previous_active,
                         # The graph needs always to be defined at init to check associated callback
                         html.Div(dcc.Graph(id='bottom-layer-figure'), hidden=True))

    # @abstract
    def get_layer_tab_content(self, active_tab: str, unit_idx=None):
        """ Get the content of the selected tab """
        return html.Div(dcc.Graph(id='bottom-layer-figure'), hidden=True)

    # @abstract
    def get_unit_title(self, unit_idx: int):
        """ Get unit description to be included in the Dash Column """
        return [html.H5(('Unit #%s' % unit_idx) +
                        (' (%s)' % self.unit_names[unit_idx] if self.unit_names is not None else ""))]

    # @abstract
    def get_unit_tabs(self, unit_idx: int, previous_active: str = None):
        """ Get the unit tab bar and layout function """
        return [*tabs.make('bottom-unit', {}, previous_active),
                # The graph needs always to be defined at init to check associated callback
                html.Div(dcc.Graph(id='bottom-unit-figure'), hidden=True)]

    # @abstract
    def get_unit_tab_content(self, unit_idx: int, active_tab: str):
        """ Get the content of the selected tab
            @return Dash HTML element (list)
        """
        return html.Div(dcc.Graph(id='bottom-unit-figure'), hidden=True)

    # @abstract
    def get_unit_description(self, unit_idx: int):
        """ Get layer Unit description to be included in a Dash Column
            @return Dash HTML element list
        """
        return [html.H5(('Unit #%s' % unit_idx) +
                        (' (%s)' % self.unit_names[unit_idx] if self.unit_names is not None else ""))]

    # @abstract
    def get_activation_map(self, activation_mapper, input_img, unit_idx):
        """ Get the activation map plot
            @return Dash HTML element list
        """
        return []

    def _get_y_offset(self):
        """ index of the first unit (lowest y) """
        return -self.num_unit * self.spacing_y / 2

    def _get_unit_labels(self, prefix='Unit '):
        """ Get series of labels corresponding to units """
        return ['%s%d' % (prefix, idx) for idx in np.arange(self.num_unit)] if self.unit_names is None \
            else self.unit_names

    @staticmethod
    def _get_graph_config():
        """" Graph config for all detail layer/unit figures """
        return dict(scrollZoom=True, displaylogo=False,
                    modeBarButtonsToRemove=['lasso2d', 'resetScale2d', 'zoomIn2d', 'zoomOut2d',
                                            'toggleSpikelines', 'select2d',
                                            'hoverClosestCartesian', 'hoverCompareCartesian'])
