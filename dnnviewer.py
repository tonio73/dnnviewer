#!/usr/bin/env python3

import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import plotly.graph_objects as go

import dnnviewerlib
import dnnviewerlib.imageutils as imageutils
import dnnviewerlib.layers
import dnnviewerlib.bridge.tensorflow as tf_bridge

import argparse

grapher = dnnviewerlib.Grapher()

parser = argparse.ArgumentParser()
parser.add_argument("-k", "--model-keras", help="Load a Keras model from file")
parser.add_argument("--test-dataset", help="Load a predefined test dataset")
parser.add_argument("--debug", help="Set Dash in debug mode")
parser.parse_args()

topn_init = 3
test_sample_init = 0
has_test_sample = False
x_test, y_test = [], []
input_classes, output_classes = None, None

args = parser.parse_args()

if args.test_dataset:
    if args.test_dataset == 'cifar-10':
        # Load test data (CIFAR-10)
        x_test, y_test, input_classes, output_classes = tf_bridge.keras_load_cifar_test_data(30, 0)
        has_test_sample = True
    elif args.test_dataset == 'mnist':
        # Load test data (MNIST)
        x_test, y_test, input_classes, output_classes = tf_bridge.keras_load_mnist_test_data(30, 0)
        has_test_sample = False

activation_mapper = None
if args.model_keras:
    # Create all other layers from the Keras Sequential model
    activation_mapper = tf_bridge.keras_load_sequential_network(grapher, args.model_keras, input_classes,
                                                                output_classes)

debug_mode = args.debug

# Create App, set stylesheets
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = 'DNN Viewer'

# Main network view
main_view = go.Figure()
main_view.update_layout(margin=dict(l=10, r=10, b=30, t=30))

grapher.plot_layers(main_view)

# Init initially shown connections based on the first test sample
if has_test_sample:
    grapher.plot_topn_connections(main_view, topn_init, grapher.layers[-1], y_test[test_sample_init])

app.layout = dbc.Container([

    # Title
    dbc.Row([
        dbc.Col([html.H1('Deep Neural Network Viewer')])
    ]),

    # Top toolbar
    dbc.Row([
        dbc.Col(html.Label('Show top n input connections:'), md=3),
        dbc.Col(dcc.Slider(id='topn-criteria',
                           min=1.0, max=4.0, step=1.0, value=topn_init,
                           marks={str(d): str(d) for d in range(0, 5)}),
                md=2)
    ]),

    # Main View
    dcc.Graph(
        id='network-view',
        figure=main_view
    ),

    # Bottom detail panel
    dbc.Row([
        dbc.Col([
            html.Div([html.Label('Select test sample'),
                      dcc.Dropdown(
                          id='select-test-sample',
                          value=test_sample_init,
                          options=[{'label': "%d (%s)" % (i, output_classes[c]), 'value': i} for i, c in
                                   enumerate(y_test)]
                      )]),
            html.P([html.Img(id='test-sample-img', alt='Sample input')],
                   className='thumbnail', hidden=not has_test_sample)
        ], md=2, align='start'),
        dbc.Col([
            html.Label('Selected layer'),
            html.Div(id='layer-info', className='detail-section')
        ], md=3, align='start'),
        dbc.Col([
            html.Label('Selected unit'),
            html.Div(id='unit-info', className='detail-section')
        ], md=4, align='start'),
        dbc.Col([html.Label('Activation maps'), html.Div(id='activation-maps')],
                md=3, align='start')
    ], style={'marginTop': '10px', 'marginBottom': '20px'})

], fluid=True)


@app.callback(Output('network-view', 'figure'),
              [Input('topn-criteria', 'value'), Input('network-view', 'clickData')])
def update_figure(topn, click_data):
    if click_data:
        point = click_data['points'][0]
        layer = grapher.layers[int(point['curveNumber'])]
        unit_idx = point['pointNumber']
        grapher.plot_topn_connections(main_view, topn, layer, unit_idx)
    return main_view


@app.callback(Output('test-sample-img', 'src'),
              [Input('select-test-sample', 'value')])
def update_test_sample(index):
    if index is not None and x_test is not None:
        img = x_test[index]
        return dnnviewerlib.imageutils.array_to_img_src(img)
    return ''


@app.callback(Output('layer-info', 'children'),
              [Input('network-view', 'clickData')])
def update_layer_info(click_data):
    if click_data:
        layer = grapher.layers[int(click_data['points'][0]['curveNumber'])]
        return layer.get_layer_description()
    return []


@app.callback(Output('unit-info', 'children'),
              [Input('network-view', 'clickData')])
def update_unit_info(click_data):
    if click_data:
        point = click_data['points'][0]
        layer = grapher.layers[int(point['curveNumber'])]
        unit_idx = point['pointNumber']
        return layer.get_unit_description(unit_idx)
    return []


@app.callback(Output('activation-maps', 'children'),
              [Input('select-test-sample', 'value'), Input('network-view', 'clickData')])
def update_activation_map(index, click_data):
    if index is not None and x_test is not None \
            and activation_mapper \
            and click_data:
        point = click_data['points'][0]
        layer = grapher.layers[int(point['curveNumber'])]
        unit_idx = point['pointNumber']
        if isinstance(layer, dnnviewerlib.layers.Convo2D):
            maps = activation_mapper.get_activation(x_test[index], layer, unit_idx)
            if unit_idx is None:
                return [html.Div(html.Img(id='activation-map', alt='Activation map',
                                          src=imageutils.array_to_img_src(imageutils.to_8bit_img(img))),
                                 className='thumbnail') for img in maps]
            else:
                return [html.H5('Unit #%s activation' % unit_idx),
                        html.Div(html.Img(id='activation-map', alt='Activation map',
                                          src=dnnviewerlib.imageutils.array_to_img_src(imageutils.to_8bit_img(maps))),
                                 className='thumbnail')]
    return []


if __name__ == '__main__':
    app.run_server(debug=debug_mode)
