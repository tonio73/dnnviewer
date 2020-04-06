from .layers.AbstractLayer import AbstractLayer
from .widgets import tabs

import plotly.graph_objects as go
import dash_core_components as dcc
import dash_html_components as html

import numpy as np
import string


class Grapher:
    """ Graph a (sequential) neural network"""

    # Number of connections to show at init
    topn_init = 3

    name = ''

    # Training properties
    training_props = {'loss': '', 'optimizer': ''}

    def __init__(self, plotly_theme='plotly_dark'):
        self.layers = []
        self.x_spacing = 1.
        self.x_offset = 0
        self.plotly_theme = plotly_theme

    def add_layer(self, layer: AbstractLayer):
        """ Add layer to graph """
        layer.set_xoffset(self.x_offset)
        self.layers.append(layer)
        self.x_offset += self.x_spacing
        return len(self.layers) - 1

    def get_layer_unit_from_click_data(self, click_data):
        point = click_data['points'][0]
        layer = self.layers[int(point['curveNumber'])]
        unit_idx = point['pointNumber']
        return layer, unit_idx

    def plot_layers(self, fig: go.Figure):
        """ Plot layers """

        layer_names = []
        for layer in self.layers:
            layer.plot(fig)
            layer_names.append(layer.name)

        # Set layer names as x axis ticks
        x_axis = dict(
            tickmode='array',
            tickvals=np.arange(len(layer_names)),
            ticktext=layer_names
        )

        fig.update_layout(showlegend=False, xaxis=x_axis, clickmode='event+select', template=self.plotly_theme)

    def plot_topn_connections(self, fig: go.Figure, topn: int, ref_layer: AbstractLayer, ref_unit: int):
        """ Add the top N connections of each neuron on the graph """

        # Plot connections as shapes (SVG paths)
        layer_index = self.layers.index(ref_layer)
        shapes = []

        # Backward from ref_layer
        sel_units = [ref_unit]
        for i in range(layer_index, 0, -1):
            cur_layer = self.layers[i]
            prev_layer = self.layers[i - 1]
            sel_units, new_shapes = cur_layer.plot_topn_connections(prev_layer, topn, sel_units, True)
            shapes[0:0] = new_shapes
            # Clip topn to 1 if number of active units is large
            if len(sel_units) > 32:
                topn = 1

        # Forward from ref_layer
        sel_units = [ref_unit]
        for i in range(layer_index + 1, len(self.layers), +1):
            cur_layer = self.layers[i]
            prev_layer = self.layers[i - 1]
            sel_units, new_shapes = cur_layer.plot_topn_connections(prev_layer, topn, sel_units, False)
            shapes[0:0] = new_shapes
            # Clip topn to 1 if number of active units is large
            if len(sel_units) > 32:
                topn = 1

        # Eventually apply the shape list
        fig.update_layout(shapes=shapes)

    def get_model_tabs(self, previous_active: string):
        """ Get the layer tab bar and layout function """
        return tabs.make('center-model', {'info': 'Info', 'config': 'Config'}, previous_active,
                         self.get_model_tab_content())

    def get_model_tab_content(self, active_tab=None):
        """ Get the content of the selected tab """
        if active_tab == 'info':
            return [html.Ul([html.Li("%s: %s" % (label, self.training_props[prop]))
                            for label, prop in zip(['Loss type', 'Optimizer type'],
                                                   list(self.training_props))])]
        elif active_tab == 'config':
            return [html.Label('Show top n connections:'),
                    dcc.Slider(id='center-topn-criteria-slider',
                               min=1.0, max=4.0, step=1.0, value=self.topn_init,
                               marks={str(d): str(d) for d in range(0, 5)})
                    ]
        return html.Div([dcc.Slider(id='center-topn-criteria-slider')], hidden=True)
