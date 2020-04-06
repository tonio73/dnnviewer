#
# Bottom pane of the application: 4 quadrants with input sample, layer, unit and activation/saliency maps
#

from .. import app, grapher, test_data, model_sequence
from ..layers.AbstractLayer import AbstractLayer

import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output

from ..imageutils import array_to_img_src
from ..widgets import activation_map

# maximum number of test sample to display in selectors
max_test_samples = 40

# Test sample to show on init
test_sample_init = 0


def render():
    return


def get_layout():
    """ Get pane layout """

    dummy_layer = AbstractLayer('dummy')

    return dbc.Row([
        # Input sample selection
        dbc.Col([
            html.Div([html.Label('Select test sample'),
                      dcc.Dropdown(
                          id='bottom-select-test-sample',
                          value=test_sample_init,
                          options=[{'label': "%d (%s)" % (i, test_data.output_classes[c]), 'value': i} for i, c in
                                   enumerate(test_data.y[:max_test_samples])]
                      )]),
            html.P([html.Img(id='bottom-test-sample-img', alt='Sample input')],
                   className='thumbnail', hidden=not test_data.has_test_sample)
        ], md=2, align='start'),

        # Layer information
        dbc.Col([
            html.Label('Selected layer'),
            html.Div(className='detail-section',
                     children=[
                         html.Div(id='bottom-layer-title'),
                         html.Div(id='bottom-layer-tabs',
                                  children=dummy_layer.get_layer_tabs(None))
                     ])
        ], md=3, align='start'),

        # Unit information
        dbc.Col([
            html.Label('Selected unit'),
            html.Div(className='detail-section',
                     children=[
                         html.Div(id='bottom-unit-title'),
                         html.Div(id='bottom-unit-tabs',
                                  children=dummy_layer.get_unit_tabs(0, None))
                     ])
        ], md=4, align='start'),

        # Activation maps
        dbc.Col([html.Label('Activation maps'),
                 html.Div(id='bottom-activation-maps')],
                md=3, align='start')
    ], style={'marginTop': '10px', 'marginBottom': '20px'})


def callbacks():
    """ Local callbacks """

    @app.callback(Output('bottom-test-sample-img', 'src'),
                  [Input('bottom-select-test-sample', 'value')])
    def update_test_sample(index):
        """ Update the display of the selected test sample upon selection
            @return the image to be displayed as base64 encoded png
        """
        if index is not None and test_data.x is not None:
            img = test_data.x[index]
            return array_to_img_src(img)
        return ''

    @app.callback(Output('bottom-activation-maps', 'children'),
                  [Input('bottom-select-test-sample', 'value'), Input('center-main-view', 'clickData')])
    def update_activation_map(index, click_data):
        if index is not None and test_data.x is not None \
                and model_sequence \
                and click_data:
            layer, unit_idx = grapher.get_layer_unit_from_click_data(click_data)
            return activation_map.widget(model_sequence, layer, unit_idx, test_data.x[index])
        return []

    return
