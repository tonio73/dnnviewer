from .AbstractLayer import AbstractLayer
from ..widgets import tabs

import plotly.graph_objects as go

import numpy as np
import logging


class Input(AbstractLayer):
    """ Simple placeholder layer to act as input to the network """
    def __init__(self, name, num_unit, plotly_theme, unit_names=None):
        if unit_names is not None and len(unit_names) != num_unit:
            logger = logging.getLogger(__name__)
            logger.error("Wrong length of input classes, got %d, expecting %d", len(unit_names), num_unit)
            unit_names = None
        AbstractLayer.__init__(self, name, 'Input', num_unit, None, plotly_theme, unit_names=unit_names)

    # @override
    def plot(self, fig):
        x, y = self.get_positions()
        hover_text = ['%d' % idx for idx in np.arange(self.num_unit)] if self.unit_names is None else self.unit_names
        fig.add_trace(go.Scatter(x=x, y=y, hovertext=hover_text, mode='markers', hoverinfo='text', name=self.name))

    # @override
    def get_layer_tabs(self, previous_active: str = None):
        """ Get the layer tab bar and layout function """
        return tabs.make('bottom-layer', {'info': 'Info'}, previous_active)

    # @override
    def get_layer_tab_content(self, active_tab, unit_idx=None):
        """ Get the content of the selected tab """
        if active_tab == 'info':
            return self._get_layer_info(), None
        return AbstractLayer.get_layer_tab_content(self, active_tab)

    # @override
    def get_unit_tabs(self, unit_idx: int, previous_active: str = None):
        """ Get the layer tab bar and layout function """
        return tabs.make('bottom-unit', {'info': 'Info'}, previous_active)

    # @override
    def get_unit_tab_content(self, unit_idx, active_tab):
        """ Get the content of the selected tab """
        w = self.weights[:, unit_idx]

        if active_tab == 'info':
            return self._get_unit_info(unit_idx, len(w)), None

        return AbstractLayer.get_layer_tab_content(self, active_tab)
