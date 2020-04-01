#
# Bottom pane of the application: 4 quadrants with input sample, layer, unit and activation/saliency maps
#

from dnnviewerapp import app, grapher, test_data
from dnnviewerapp.layers.AbstractLayer import AbstractLayer

import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output

import dnnviewerapp.imageutils as imageutils
from dnnviewerapp.widgets import activation_map

# maximum number of test sample to display in selectors
max_test_samples = 40


def render():
    return


def get_layout():
    """ Get pane layout """
    return dbc.Row([
        # Input sample selection
        dbc.Col([
            html.Div([html.Label('Select test sample'),
                      dcc.Dropdown(
                          id='select-test-sample',
                          value=test_data.test_sample_init,
                          options=[{'label': "%d (%s)" % (i, test_data.output_classes[c]), 'value': i} for i, c in
                                   enumerate(test_data.y_test[:max_test_samples])]
                      )]),
            html.P([html.Img(id='test-sample-img', alt='Sample input')],
                   className='thumbnail', hidden=not test_data.has_test_sample)
        ], md=2, align='start'),

        # Layer information
        dbc.Col([
            html.Label('Selected layer'),
            html.Div(className='detail-section',
                     children=[
                         html.Div(id='layer-title'),
                         html.Div(id='layer-tabs',
                                  children=[*AbstractLayer.make_tabs('layer', {}, None), dcc.Graph(id='layer-figure')])
                     ])
        ], md=3, align='start'),

        # Unit information
        dbc.Col([
            html.Label('Selected unit'),
            html.Div(className='detail-section',
                     children=[
                         html.Div(id='unit-title'),
                         html.Div(id='unit-tabs',
                                  children=[*AbstractLayer.make_tabs('unit', {}, None), dcc.Graph(id='unit-figure')])
                     ])
        ], md=4, align='start'),

        # Activation maps
        dbc.Col([html.Label('Activation maps'),
                 html.Div(id='activation-maps')],
                md=3, align='start')
    ], style={'marginTop': '10px', 'marginBottom': '20px'})


def callbacks():
    """ Local callbacks """

    @app.callback(Output('test-sample-img', 'src'),
                  [Input('select-test-sample', 'value')])
    def update_test_sample(index):
        """ Update the display of the selected test sample upon selection
            @return the image to be displayed as base64 encoded png
        """
        if index is not None and test_data.x_test is not None:
            img = test_data.x_test[index]
            return imageutils.array_to_img_src(img)
        return ''

    @app.callback(Output('activation-maps', 'children'),
                  [Input('select-test-sample', 'value'), Input('network-view', 'clickData')])
    def update_activation_map(index, click_data):
        if index is not None and test_data.x_test is not None \
                and grapher.activation_mapper \
                and click_data:
            layer, unit_idx = grapher.get_layer_unit_from_click_data(click_data)
            return activation_map.widget(grapher.activation_mapper, layer, unit_idx, test_data.x_test[index])
        return []

    return
