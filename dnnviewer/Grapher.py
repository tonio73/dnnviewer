from .layers.AbstractLayer import AbstractLayer
from .layers import Convo2D, Dense
from .widgets import tabs, property_list
from .theming.Theme import Theme

import plotly.graph_objects as go
import dash_core_components as dcc
import dash_html_components as html

import numpy as np
from typing import List
import logging


class Grapher:
    """ Graph a (sequential) neural network"""

    # Number of connections to show at init
    topn_init = 3

    layers: List[AbstractLayer]

    def __init__(self, theme=Theme()):
        self.name = ''
        self.layers = []
        self.x_spacing = 1.
        self.x_offset = 0
        self.structure_props = {'num_dense': 0, 'num_convo2d': 0}
        self.training_props = {'loss': '', 'optimizer': ''}
        self.theme = theme

    def clear_layers(self):
        """ Reset layers """
        self.layers = []
        self.structure_props['num_dense'] = 0
        self.structure_props['num_convo2d'] = 0
        self.x_offset = 0

    def add_layer(self, layer: AbstractLayer):
        """ Add layer to graph """
        layer.set_xoffset(self.x_offset)
        self.layers.append(layer)
        self.x_offset += self.x_spacing
        if isinstance(layer, Dense.Dense):
            self.structure_props['num_dense'] += 1
        elif isinstance(layer, Convo2D.Convo2D):
            self.structure_props['num_convo2d'] += 1
        return len(self.layers) - 1

    def get_layer_unit_from_click_data(self, click_data):
        point = click_data['points'][0]
        layer = self.layers[int(point['curveNumber'])]
        unit_idx = point['pointNumber']
        return layer, unit_idx

    def plot_layers(self, fig: go.Figure):
        """ Plot layers """

        layer_names = []
        layer: AbstractLayer
        for layer in self.layers:
            layer.plot(fig)
            layer_names.append(layer.name)

        # Set layer names as x axis ticks
        x_axis = dict(
            tickmode='array',
            tickvals=np.arange(len(layer_names)),
            ticktext=layer_names
        )

        fig.update_layout(showlegend=False, xaxis=x_axis, clickmode='event+select', template=self.theme.plotly)

    def plot_topn_connections(self, fig: go.Figure, topn: int, ref_layer_idx: int, ref_unit: int):
        """ Add the top N connections of each neuron on the graph """

        logger = logging.getLogger(__name__)

        logger.debug('plot_topn_connections, layer=%d, unit=%d', ref_layer_idx, ref_unit)

        assert 0 <= ref_layer_idx < len(self.layers), 'Attempting to reach layer %d while number of layer is %d' % \
                                                      (ref_layer_idx, len(self.layers))

        # Plot connections as shapes (SVG paths)
        shapes = []

        # Backward from ref_layer
        sel_units = [ref_unit]
        for i in range(ref_layer_idx, 0, -1):
            cur_layer = self.layers[i]
            prev_layer = self.layers[i - 1]
            sel_units, new_shapes = cur_layer.plot_topn_connections(prev_layer, topn, sel_units, True)
            shapes[0:0] = new_shapes
            # Clip topn to 1 if number of active units is large
            if len(sel_units) > 32:
                topn = 1

        # Forward from ref_layer
        sel_units = [ref_unit]
        for i in range(ref_layer_idx + 1, len(self.layers), +1):
            cur_layer = self.layers[i]
            prev_layer = self.layers[i - 1]
            sel_units, new_shapes = cur_layer.plot_topn_connections(prev_layer, topn, sel_units, False)
            shapes[0:0] = new_shapes
            # Clip topn to 1 if number of active units is large
            if len(sel_units) > 32:
                topn = 1

        # Annotation on the selected layer (trace)
        sel_layer: AbstractLayer = self.layers[ref_layer_idx]
        sel_pos = sel_layer.get_unit_position(ref_unit, True)
        annotations = [dict(x=sel_pos[0], y=sel_pos[1],
                            xref="x", yref="y",
                            text="%s #%d" % (sel_layer.name, ref_unit),
                            showarrow=True, arrowhead=3, arrowcolor="#aaa",
                            bgcolor="#666", borderpad=4,
                            ax=-(10 + 8 * len(sel_layer.name)), ay=-30)]

        # Eventually apply the shape list
        fig.update_layout(shapes=shapes, annotations=annotations)

    def get_model_tabs(self, previous_active: str):
        """ Get the layer tab bar and layout function """
        return tabs.make('center-model', {'info': 'Info', 'config': 'Config'}, previous_active,
                         self.get_model_tab_content())

    def get_model_tab_content(self, active_tab=None):
        """ Get the content of the selected tab """
        if active_tab == 'info':
            return [*property_list.widget('model_structure', 'Structure',
                                          Grapher._structure_properties_labels, self.structure_props),
                    *property_list.widget('model_training', 'Training',
                                          Grapher._training_properties_labels, self.training_props)]
        elif active_tab == 'config':
            return [html.Label('Show top n connections:'),
                    dcc.Slider(id='center-topn-criteria-slider',
                               min=1.0, max=4.0, step=1.0, value=self.topn_init,
                               marks={str(d): str(d) for d in range(0, 5)})
                    ]
        return html.Div([dcc.Slider(id='center-topn-criteria-slider')], hidden=True)

    _structure_properties_labels = dict(num_dense="Dense layers", num_convo2d='Convolutional 2D')
    _training_properties_labels = dict(loss="Loss", optimizer='Optimizer')
