from ..widgets import tabs, property_list
from ..theming.Theme import Theme, float_fmt

import numpy as np
import plotly.graph_objects as go
import dash_html_components as html


class AbstractLayer:
    """ Abstract layer representation in Viewer """

    def __init__(self, name, layer_type='unknown', num_unit=0, weights=None, grads=None,
                 theme: Theme = Theme(),
                 unit_names=None):
        self.name = name
        self.num_unit = num_unit
        self.weights = weights
        self.bias = None
        self.grads = grads
        self.unit_names = unit_names
        self.structure_props = {'type': layer_type, 'num_unit': num_unit}
        self.training_props = {}
        self.input_props = {}
        self.output_props = {}
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
        return html.H5("Layer '%s'" % self.name)

    # @abstract
    def get_layer_tabs(self, previous_active: str = None):
        """ Get the layer tab bar and layout function """
        return tabs.make('bottom-layer', {}, previous_active)

    # @abstract
    def get_layer_tab_content(self, active_tab: str, unit_idx=None):
        """ Get the content of the selected tab
            @return duple: content (HTML) and figure (or None if no figure)
        """
        return [], None

    # @abstract
    def get_unit_title(self, unit_idx: int):
        """ Get unit description to be included in the Dash Column """
        return [html.H5(('Unit #%s' % unit_idx) +
                        (' (%s)' % self.unit_names[unit_idx] if self.unit_names is not None else ""))]

    # @abstract
    def get_unit_tabs(self, unit_idx: int, previous_active: str = None):
        """ Get the unit tab bar and layout function """
        return tabs.make('bottom-unit', {}, previous_active)

    # @abstract
    def get_unit_tab_content(self, unit_idx: int, active_tab: str):
        """ Get the content of the selected tab
            @return duple: content (HTML) and figure (or None if no figure)
        """
        return [], None

    # @abstract
    def get_activation_map(self, activation_mapper, input_img, unit_idx):
        """ Get the activation map plot
            @return duple: content (HTML) and figure (or None if no figure)
        """
        return [], None

    def _get_y_offset(self):
        """ index of the first unit (lowest y) """
        return -self.num_unit * self.spacing_y / 2

    def _get_unit_labels(self, prefix='Unit '):
        """ Get series of labels corresponding to units """
        return ['%s%d' % (prefix, idx) for idx in np.arange(self.num_unit)] if self.unit_names is None \
            else self.unit_names

    def _get_layer_info(self):
        """ Build HTML components for the layer information (tab) """
        return [*property_list.widget('layer_structure', 'Structure',
                                      _structure_properties_labels, self.structure_props),
                *property_list.widget('layer_input', 'Input', _input_properties_labels, self.input_props),
                *property_list.widget('layer_output', 'Output', _output_properties_labels, self.output_props),
                *property_list.widget('layer_training', 'Training', _training_properties_labels, self.training_props)]

    def _get_unit_info(self, unit_idx, num_coef):
        """ Build HTML components for the unit information (tab) """
        unit_structure_props = {'num_coef': num_coef}
        unit_training_props = {}
        if self.bias is not None:
            unit_training_props['bias'] = float_fmt % self.bias[unit_idx]
        return [*property_list.widget('unit_structure', 'Structure',
                                      _unit_structure_properties_labels, unit_structure_props),
                *property_list.widget('unit_training', 'Training',
                                      _unit_training_properties_labels, unit_training_props)]


# Property labels
_structure_properties_labels = {'type': 'Layer type',
                                'num_unit': "Num units",
                                'strides': "Strides",
                                'padding': 'Padding',
                                'activation': 'Activation'}
_training_properties_labels = {'dropout': "Drop out (at input)",
                               'activity_regularizer': "Activity reg.",
                               'kernel_regularizer': "Kernel reg.",
                               'bias_regularizer': "Bias reg."}
_input_properties_labels = {'shape': "Shape",
                            'batch_normalization': "Batch normalization"}
_output_properties_labels = {'shape': "Shape",
                             'pooling': "Pooling"}

_unit_structure_properties_labels = {'num_coef': 'Num coefficients'}
_unit_training_properties_labels = {'bias': 'Bias'}
