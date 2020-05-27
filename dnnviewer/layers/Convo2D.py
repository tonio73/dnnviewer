from .AbstractLayer import AbstractLayer
from ..Connector import Connector
from ..Statistics import Statistics
from ..bridge.AbstractActivationMapper import AbstractActivationMapper
from ..theming.Theme import Theme
from ..widgets import layer_minimax_graph, tabs, conv_filter_map
from ..imageutils import array_to_img_src, to_8bit_img

import plotly.graph_objects as go
import dash_html_components as html

import numpy as np


class Convo2D(AbstractLayer):
    """ Convolutional layer of n units """
    """ Assume 4D weight tensor with dimensions: filter_x, filter_y, input_filter, output_filter """

    def __init__(self, name, num_unit, weights, grads, theme=Theme(),
                 unit_names=None, flatten_output=False):

        assert weights.ndim == 4
        assert num_unit == weights.shape[3]

        AbstractLayer.__init__(self, name, 'Convolutional 2D', num_unit, weights, grads, theme, unit_names)

        self.flatten_output = flatten_output

    # @override
    def get_unit_position(self, unit_idx, at_output=False):
        """ Get single or vector of unit positions """
        if isinstance(unit_idx, int):
            x = self.xoffset
        else:
            x = self.xoffset * np.ones(len(unit_idx))

        # If output is flattened, actual output dimension is larger than number of filters => wrap with modulo
        if at_output and self.flatten_output:
            return x, self._get_y_offset() + self.spacing_y * (unit_idx % self.num_unit)
        else:
            return x, self._get_y_offset() + self.spacing_y * unit_idx

    # @override
    def plot(self, fig):
        x, y = self.get_positions()
        hover_text = self._get_unit_labels()
        fig.add_trace(go.Scatter(x=x, y=y, hovertext=hover_text, mode='markers', hoverinfo='text', name=self.name))

    # @override
    def plot_topn_connections(self, backward_layer, topn, active_units, backward):

        """ On convolution layers, the maximum over 2D convolution filters is first extracted,
            then take the top n across the (input, output) filter pairs
        """

        if self.weights is None:
            return np.zeros(0), []

        # Issues with flatten_output
        # assert backward_layer.num_unit == self.weights.shape[2]

        if backward:

            # Active units need to wrapped if the output is flattened
            if self.flatten_output:
                active_units = np.mod(active_units, self.num_unit)

            # Max on the 2D convolution filters
            weights1 = self.weights.reshape(-1, backward_layer.num_unit, self.num_unit)
            convo_max_weights1 = weights1.take(np.argmax(np.abs(weights1[:, :, active_units]), axis=0))

            # Top N on the (input, output) pair
            strongest_idx, strongest = Statistics.get_strongest(convo_max_weights1,
                                                                min(topn, backward_layer.num_unit))
            # For each of the top n, create a vector of connectors and plot it
            to_indexes = np.tile(active_units, strongest.shape[0])

            strongest_idx = strongest_idx.ravel()
            strongest = strongest.ravel()

            connectors = Connector(backward_layer, self,
                                   strongest_idx, to_indexes, strongest,
                                   self.theme.weight_color_scale)

        else:
            # Max on the 2D convolution filters
            # Transpose here
            weights1 = np.swapaxes(self.weights, 2, 3).reshape((-1, self.num_unit, backward_layer.num_unit))
            convo_max_weights1 = weights1.take(np.argmax(np.abs(weights1[:, :, active_units]), axis=0))  # <--

            # No need to handle the flatten as active_units is already wrapped

            strongest_idx, strongest = Statistics.get_strongest(convo_max_weights1,
                                                                min(topn, backward_layer.num_unit))

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
            weights1 = self.weights.reshape(-1, self.weights.shape[3])
            fig = layer_minimax_graph.figure(weights1, self.num_unit, self.unit_names,
                                             self.theme, self.theme.weight_color_scale, unit_idx)
            return [], fig

        elif active_tab == 'grads':
            if self.grads is None:
                return html.P("No gradients available"), None

            grads1 = self.grads.reshape(-1, self.grads.shape[3])
            fig = layer_minimax_graph.figure(grads1, self.num_unit, self.unit_names,
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
        w = self.weights[:, :, :, unit_idx]

        if active_tab == 'info':
            return self._get_unit_info(unit_idx, len(w.ravel())), None

        elif active_tab == 'weights':
            fig = conv_filter_map.figure(w, self.theme, self.theme.weight_color_scale)
            return [], fig

        elif active_tab == 'grads':
            if self.grads is None:
                return html.P("No gradients available"), None

            fig = conv_filter_map.figure(self.grads[:, :, :, unit_idx], self.theme, self.theme.gradient_color_scale)
            return [], fig

        return AbstractLayer.get_layer_tab_content(self, active_tab)

    def get_activation_map(self, activation_mapper: AbstractActivationMapper, input_img, unit_idx):
        """ Get the activation map plot """

        maps = activation_mapper.get_activation(input_img, self, unit_idx)
        # Images are wrapped in Img HTML element => no Plotly GO figure
        if unit_idx is None:
            return [html.Div(html.Img(id='activation-map', alt='Activation map',
                                      src=array_to_img_src(to_8bit_img(img))),
                             className='thumbnail') for img in maps], None
        else:
            return html.Div(html.Img(id='activation-map', alt='Activation map',
                                     src=array_to_img_src(to_8bit_img(maps))),
                            className='thumbnail'), None
