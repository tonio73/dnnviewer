from .AbstractLayer import AbstractLayer
from ..bridge.AbstractActivationMapper import AbstractActivationMapper
from ..widgets import tabs
from ..imageutils import array_to_img_src, to_8bit_img
from ..Connector import Connector
from ..utils import array

import plotly.graph_objects as go
import dash_html_components as html

import numpy as np
import logging


class Reshape(AbstractLayer):
    """ Simple placeholder layer to act as input to the network """
    def __init__(self, name: str, path: str, input_shape, output_shape,
                 plotly_theme, unit_names=None):
        self.input_shape = input_shape
        self.output_shape = output_shape
        num_unit = output_shape[-1]
        if unit_names is not None and len(unit_names) != num_unit:
            logger = logging.getLogger(__name__)
            logger.error("Wrong length of input classes, got %d, expecting %d", len(unit_names), num_unit)
            unit_names = None
        AbstractLayer.__init__(self, name, path, 'Reshape', num_unit, None, plotly_theme, unit_names=unit_names)

    # @override
    def plot(self, fig):
        x, y = self.get_positions()
        hover_text = ['%d' % idx for idx in np.arange(self.num_unit)] if self.unit_names is None else self.unit_names
        fig.add_trace(go.Scatter(x=x, y=y, hovertext=hover_text, mode='markers', hoverinfo='text', name=self.name))

    # @override
    def plot_topn_connections_backward(self, backward_layer, topn, active_units):

        out_active_units = np.zeros(len(active_units), dtype=int)
        n_dim_out = len(self.output_shape)
        for i, unit in enumerate(active_units):
            index = np.zeros(n_dim_out)
            index[-1] = unit
            out_active_units[i] = array.multi_to_multi_dim(self.output_shape, self.input_shape, index)[-1]

        connectors = Connector(backward_layer, self,
                               out_active_units, active_units, np.ones(len(active_units)),
                               self.theme.weight_color_scale)

        return np.unique(out_active_units), connectors.get_shapes()

    # @override
    def plot_topn_connections_forward(self, backward_layer, topn, active_units):

        out_active_units = np.zeros(len(active_units), dtype=int)
        n_dim_out = len(self.output_shape)
        for i, unit in enumerate(active_units):
            index = np.zeros(n_dim_out)
            index[-1] = unit
            out_active_units[i] = array.multi_to_multi_dim(self.input_shape, self.output_shape, index)[-1]

        connectors = Connector(backward_layer, self,
                               active_units, out_active_units, np.ones(len(active_units)),
                               self.theme.weight_color_scale)

        return np.unique(out_active_units), connectors.get_shapes()

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

        if active_tab == 'info':
            return self._get_unit_info(unit_idx, 0), None

        return AbstractLayer.get_layer_tab_content(self, active_tab)

    def get_activation_map(self, activation_mapper: AbstractActivationMapper, input_img, unit_idx):
        """ Get the activation map plot """

        maps = activation_mapper.get_activation(input_img, self, unit_idx)
        if maps is None:
            return [], None

        # Images are wrapped in Img HTML element => no Plotly GO figure
        if unit_idx is None:
            if len(self.output_shape) == 3:
                # Display as an image (similar to Convo2D)
                return [html.Div(html.Img(id='activation-map', alt='Activation map',
                                          src=array_to_img_src(to_8bit_img(img))),
                                 className='thumbnail') for img in maps], None
        else:
            if len(self.output_shape) == 2:
                # Display as a graph (similar to Dense)
                hover_text = self._get_unit_labels()
                fig = go.Figure(data=[go.Bar(x=maps, hovertext=hover_text, hoverinfo='text',
                                             marker=self.theme.activation_color_scale.as_dict(maps))])
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
                fig.update_layout(margin=self.theme.bottom_figure_margins,
                                  title=dict(text='Layer activation', font=dict(size=14)),
                                  xaxis_title_text='Amplitude',
                                  yaxis_title_text='Unit',
                                  bargap=0.2,  # gap between bars of adjacent location coordinates)
                                  template=self.theme.plotly,
                                  annotations=annotations)

            elif len(self.output_shape) == 3:
                return html.Div(html.Img(id='activation-map', alt='Activation map',
                                         src=array_to_img_src(to_8bit_img(maps))),
                                className='thumbnail'), None

        return [], None
