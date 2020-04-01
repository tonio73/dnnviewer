#!/usr/bin/env python3

import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State

from dnnviewerapp import grapher, app, test_data
from dnnviewerapp.panes import top, center, bottom
from dnnviewerapp.layers.AbstractLayer import AbstractLayer
import dnnviewerapp.bridge.tensorflow as tf_bridge

import argparse

panes = [top, center, bottom]

parser = argparse.ArgumentParser()
parser.add_argument("-k", "--model-keras", help="Load a Keras model from file")
parser.add_argument("--test-dataset", help="Load a predefined test dataset")
parser.add_argument("--debug", help="Set Dash in debug mode")
parser.parse_args()

# Handle command line arguments
args = parser.parse_args()

if args.test_dataset:
    if args.test_dataset == 'cifar-10':
        # Load test data (CIFAR-10)
        tf_bridge.keras_load_cifar_test_data(test_data)
        has_test_sample = True
    elif args.test_dataset == 'mnist':
        # Load test data (MNIST)
        tf_bridge.keras_load_mnist_test_data(test_data)

if args.model_keras:
    # Create all other layers from the Keras Sequential model
    grapher.activation_mapper = tf_bridge.keras_load_sequential_network(grapher, args.model_keras,
                                                                        test_data.input_classes,
                                                                        test_data.output_classes)

# Prepare rendering of panes
[pane.render() for pane in panes]


# Top app layout
app.layout = dbc.Container([

    *[pane.get_layout() for pane in panes]

], fluid=True)


# Global callbacks


@app.callback(Output('layer-title', 'children'),
              [Input('network-view', 'clickData')])
def update_layer_info(click_data):
    """ Display selected layer title """
    if click_data:
        layer, unit_idx = grapher.get_layer_unit_from_click_data(click_data)
        return layer.get_layer_title()
    return []


@app.callback(Output('layer-tabs', 'children'),
              [Input('network-view', 'clickData')],
              [State('layer-tab-bar', 'active_tab')])
def update_layer_tabs(click_data, active_tab):
    if click_data:
        layer, unit_idx = grapher.get_layer_unit_from_click_data(click_data)
        return layer.get_layer_tabs(active_tab)
    return AbstractLayer.make_tabs('layer', {}, active_tab)


@app.callback(
    Output("layer-tab-content", "children"),
    [Input("layer-tab-bar", "active_tab")],
    [State('network-view', 'clickData')])
def render_layer_tab_content(active_tab, network_click_data):
    """ layer info tab selected => update content """
    if active_tab and network_click_data is not None:
        layer, unit_idx = grapher.get_layer_unit_from_click_data(network_click_data)
        if layer is not None:
            return layer.get_layer_tab_content(active_tab)
    return html.Div()


@app.callback(Output('unit-title', 'children'),
              [Input('network-view', 'clickData')])
def update_layer_info(click_data):
    """ Display selected unit title """
    if click_data:
        layer, unit_idx = grapher.get_layer_unit_from_click_data(click_data)
        return layer.get_unit_title(unit_idx)
    return []


@app.callback(Output('unit-tabs', 'children'),
              [Input('network-view', 'clickData')],
              [State('unit-tab-bar', 'active_tab')])
def update_unit_tabs(click_data, active_tab):
    if click_data:
        layer, unit_idx = grapher.get_layer_unit_from_click_data(click_data)
        return layer.get_unit_tabs(unit_idx, active_tab)
    return AbstractLayer.make_tabs('unit', {}, active_tab)


@app.callback(
    Output("unit-tab-content", "children"),
    [Input("unit-tab-bar", "active_tab")],
    [State('network-view', 'clickData')])
def render_unit_tab_content(active_tab, network_click_data):
    """ layer info tab selected => update content """
    if active_tab and network_click_data is not None:
        layer, unit_idx = grapher.get_layer_unit_from_click_data(network_click_data)
        if layer is not None:
            return layer.get_unit_tab_content(unit_idx, active_tab)
    return html.Div()


@app.callback(Output('network-view', 'clickData'),
              [Input('layer-figure', 'clickData')],
              [State('network-view', 'clickData')])
def update_unit_selection(click_data, network_click_data):
    """ Click on the layer figure => update the main selection's unit """
    if click_data and network_click_data:
        network_click_data['points'][0]['pointNumber'] = click_data['points'][0]['pointNumber']
    return network_click_data

# Local callbacks


[pane.callbacks() for pane in panes]


if __name__ == '__main__':
    app.run_server(debug=args.debug)
