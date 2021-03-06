from .AbstractLayer import AbstractLayer
from ..Connector import Connector
from ..utils import statistics
from ..theming.Theme import Theme
from ..widgets import layer_minimax_graph, tabs
from ..bridge.AbstractActivationMapper import AbstractActivationMapper

import numpy as np
import plotly.graph_objects as go
import dash_html_components as html


class Dense(AbstractLayer):
    """ Dense (aka fully connected) layer of n units """
    """ Assume 2D weight tensor with dimensions: previous layer unit, self unit """

    def __init__(self, name: str, path: str, num_unit, weights, grads, theme=Theme(), unit_names=None):
        assert weights.ndim == 2
        assert num_unit == weights.shape[1]

        AbstractLayer.__init__(self, name, path, 'Dense', num_unit, weights, grads, theme, unit_names)

    # @override
    def plot(self, fig):
        x, y = self.get_positions()
        hover_text = ['%d' % idx for idx in np.arange(self.num_unit)] if self.unit_names is None else self.unit_names
        fig.add_trace(go.Scatter(x=x, y=y, hovertext=hover_text, mode='markers', hoverinfo='text', name=self.name))

    # @override
    def plot_topn_connections_backward(self, backward_layer, topn, active_units):
        if self.weights is None:
            return np.empty(0), []

        strongest_idx, strongest = statistics.get_strongest(self.weights[:, active_units],
                                                            min(topn, backward_layer.num_unit))
        # For each of the top n, create a vector of connectors and plot it
        to_indexes = np.tile(active_units, strongest.shape[0])

        strongest_idx = strongest_idx.ravel()
        strongest = strongest.ravel()

        # Get actual unit indexes on previous layer
        strongest_idx = backward_layer.get_unit_index(strongest_idx, AbstractLayer.FROM_OUTPUT)

        connectors = Connector(backward_layer, self,
                               strongest_idx, to_indexes, strongest,
                               self.theme.weight_color_scale)

        return np.unique(strongest_idx), connectors.get_shapes()

    # @override
    def plot_topn_connections_forward(self, backward_layer, topn, active_units):
        if self.weights is None:
            return np.empty(0), []

        # Map in my input domain take into account for output flatten and sampling in previous layer
        sel_units = backward_layer.get_unit_index(active_units, AbstractLayer.AT_OUTPUT)

        strongest_idx, strongest = statistics.get_strongest(self.weights.T[:, sel_units],  # <-- Transpose
                                                            min(topn, self.num_unit))

        # For each of the top n, create a vector of connectors and plot it
        from_indexes = np.tile(active_units, strongest.shape[0])

        strongest_idx = strongest_idx.ravel()
        strongest = strongest.ravel()

        connectors = Connector(backward_layer, self,
                               from_indexes, strongest_idx, strongest,
                               self.theme.weight_color_scale)

        return np.unique(strongest_idx), connectors.get_shapes()

    # @override
    def get_layer_tabs(self, previous_active: str = None):
        """ Get the layer tab bar and layout function """
        return tabs.make('bottom-layer', {'info': 'Info', 'weights': 'Weights', 'grads': 'Gradients'}, previous_active)

    # @override
    def get_layer_tab_content(self, active_tab: str, unit_idx=None):
        """ Get the content of the selected tab """

        if active_tab == 'info':
            return self._get_layer_info(), None

        elif active_tab == 'weights':
            fig = layer_minimax_graph.figure(self.weights, self.num_unit, self.unit_names,
                                             self.theme, self.theme.weight_color_scale, unit_idx)
            return [], fig

        elif active_tab == 'grads':
            if self.grads is None:
                return html.P("No gradients available"), None

            fig = layer_minimax_graph.figure(self.grads, self.num_unit, self.unit_names,
                                             self.theme, self.theme.gradient_color_scale, unit_idx)
            return [], fig

        return AbstractLayer.get_layer_tab_content(self, active_tab)

    # @override
    def get_unit_tabs(self, unit_idx: int, previous_active: str = None):
        """ Get the layer tab bar and layout function """
        return tabs.make('bottom-unit', {'info': 'Info', 'weights': 'Weights', 'grads': 'Gradients'}, previous_active)

    # @override
    def get_unit_tab_content(self, unit_idx, active_tab):
        """ Get the content of the selected tab """
        w = self.weights[:, unit_idx]

        if active_tab == 'info':
            return self._get_unit_info(unit_idx, len(w)), None

        elif active_tab == 'weights':
            fig = go.Figure(data=[go.Histogram(x=w, marker=self.theme.weight_color_scale.as_dict(w))])
            fig.update_layout(margin=self.theme.bottom_figure_margins,
                              title=dict(text='Weight histogram', font=dict(size=14)),
                              xaxis_title_text='Amplitude',
                              bargap=0.2,  # gap between bars of adjacent location coordinates)
                              template=self.theme.plotly)
            return [], fig

        elif active_tab == 'grads':
            if self.grads is None:
                return html.P("No gradients available"), None

            g = self.grads[:, unit_idx]
            fig = go.Figure(data=[go.Histogram(x=g, marker=self.theme.gradient_color_scale.as_dict(g))])
            fig.update_layout(margin=self.theme.bottom_figure_margins,
                              title=dict(text='Gradients histogram', font=dict(size=14)),
                              xaxis_title_text='Amplitude',
                              # yaxis_title_text='Count',
                              bargap=0.2,  # gap between bars of adjacent location coordinates)
                              template=self.theme.plotly)
            return [], fig

        return AbstractLayer.get_layer_tab_content(self, active_tab)

    def get_activation_map(self, activation_mapper: AbstractActivationMapper, input_img, unit_idx):
        """ Get the activation map plot """

        maps = activation_mapper.get_activation(input_img, self)
        if maps is None:
            return [], None

        if self.num_unit < 64:
            trace = go.Bar(x=maps, hovertext=self._get_unit_labels(), hoverinfo='text',
                           marker=self.theme.activation_color_scale.as_dict(maps))
        else:
            trace = go.Scatter(x=maps, hovertext=self._get_unit_labels(), hoverinfo='text',
                               mode='lines', marker=self.theme.activation_color_scale.as_dict(maps))
        fig = go.Figure(data=[trace])

        if unit_idx is not None and unit_idx < self.num_unit:
            if self.unit_names:
                annotation_title = ("#%d (%s): %.3g" % (unit_idx, self.unit_names[unit_idx], maps[unit_idx]))
            else:
                annotation_title = "#%d: %.3g" % (unit_idx, maps[unit_idx])
            annotations = [dict(x=maps[unit_idx], y=unit_idx,
                                xref="x", yref="y",
                                text=annotation_title,
                                showarrow=True, arrowhead=3, arrowcolor="#aaa",
                                bgcolor="#666", borderpad=4,
                                ax=-30, ay=-30)]
        else:
            annotations = []

        fig.update_layout(margin=self.theme.bottom_figure_margins,
                          title=dict(text='Layer activation', font=dict(size=14)),
                          xaxis_title_text='Amplitude',
                          yaxis_title_text='Unit',
                          bargap=0.2,  # gap between bars of adjacent location coordinates)
                          template=self.theme.plotly,
                          annotations=annotations)

        return [], fig
