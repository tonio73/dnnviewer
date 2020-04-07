#!/usr/bin/env python3

import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State

from dnnviewerapp import grapher, app, test_data, model_sequence
from dnnviewerapp.panes import top, center, bottom
from dnnviewerapp.layers.AbstractLayer import AbstractLayer
import dnnviewerapp.bridge.tensorflow as tf_bridge

import argparse

panes = [top.TopPane(), center.CenterPane(), bottom.BottomPane()]

parser = argparse.ArgumentParser()
parser.add_argument("-k", "--model-keras", help="Load a Keras model from file")
parser.add_argument("--sequence-keras",
                    help="Load sequence of Keras checkpoints following the pattern 'dirpath/model_prefix{epoch}")
parser.add_argument("--test-dataset", help="Load a predefined test dataset (mnist, fashion-mnist, cifar-10)")
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
    elif args.test_dataset == 'fashion-mnist':
        # Load test data (Fashion MNIST)
        tf_bridge.keras_load_mnistfashion_test_data(test_data)

if args.model_keras:
    # Create all other layers from the Keras Sequential model
    model_sequence.load_single(args.model_keras)
elif args.sequence_keras:
    model_sequence.load_sequence(args.sequence_keras)

model_sequence.first_epoch(grapher)

# Prepare rendering of panes
[pane.render() for pane in panes]

# Top app layout
app.layout = dbc.Container([

    *[pane.get_layout() for pane in panes]

], fluid=True)


# Global callbacks


@app.callback([Output('bottom-layer-title', 'children'), Output('bottom-layer-tabs', 'children'),
               Output('bottom-unit-title', 'children'), Output('bottom-unit-tabs', 'children')],
              [Input('center-main-view', 'clickData'), Input('top-epoch-index', 'data')],
              [State('bottom-layer-tab-bar', 'active_tab'),
               State('bottom-unit-tab-bar', 'active_tab')])
def update_layer_info(click_data, _, active_layer_tab, active_unit_tab):
    """ Display selected layer title """
    if click_data:
        layer, unit_idx = grapher.get_layer_unit_from_click_data(click_data)
        return layer.get_layer_title(), layer.get_layer_tabs(active_layer_tab),\
            layer.get_unit_title(unit_idx), layer.get_unit_tabs(unit_idx, active_unit_tab)
    dummy_layer = AbstractLayer('dummy')
    return [], dummy_layer.get_layer_tabs(active_layer_tab), [], dummy_layer.get_unit_tabs(0, active_unit_tab)


@app.callback(Output('center-main-view', 'clickData'),
              [Input('bottom-layer-figure', 'clickData')],
              [State('center-main-view', 'clickData')])
def update_unit_selection(click_data, network_click_data):
    """ Click on the layer figure => update the main selection's unit """
    if click_data and network_click_data:
        network_click_data['points'][0]['pointNumber'] = click_data['points'][0]['pointNumber']
    return network_click_data


# Local callbacks
[pane.callbacks() for pane in panes]

if __name__ == '__main__':
    app.run_server(debug=args.debug)
