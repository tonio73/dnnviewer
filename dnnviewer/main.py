#!/usr/bin/env python3

#
# Python application main entry point
#

from .main_network_view import MainNetworkView
from .TestData import TestData
from .bridge import tensorflow as tf_bridge
from .bridge.KerasModelSequence import KerasModelSequence
from .widgets import font_awesome

import dash
import dash_bootstrap_components as dbc
import flask

import argparse


def parse_arguments():
    """ Parse command line arguments """

    # Command line options
    parser = argparse.ArgumentParser()
    parser_group = parser.add_mutually_exclusive_group(required=True)
    parser_group.add_argument("--model-keras", "-m", help="Load a Keras model from file")
    parser_group.add_argument("--sequence-keras", "-s",
                              help="Load sequence of Keras checkpoints following the pattern "
                                   "'dirpath/model_prefix{epoch}'")
    parser.add_argument("--test-dataset", "-t", help="Load a predefined test dataset (mnist, fashion-mnist, cifar-10)")
    parser.add_argument("--debug", help="Set Dash in debug mode", dest="debug", default=False, action='store_true')
    parser.parse_args()

    # Handle command line arguments
    return parser.parse_args()


def run_app(args):
    """ Run the app """

    # Create App, set stylesheets
    app = dash.Dash(__name__,
                    assets_folder="assets",
                    external_stylesheets=[dbc.themes.BOOTSTRAP, font_awesome.CDN_CSS_URL])
    app.title = 'DNN Viewer'

    # Manager for test data
    test_data = TestData()

    # Model sequence : currently only supporting from Keras
    model_sequence = KerasModelSequence(test_data)

    # Test data
    if args.test_dataset:
        if args.test_dataset == 'cifar-10':
            # Load test data (CIFAR-10)
            tf_bridge.keras_load_cifar_test_data(test_data)
        elif args.test_dataset == 'mnist':
            # Load test data (MNIST)
            tf_bridge.keras_load_mnist_test_data(test_data)
        elif args.test_dataset == 'fashion-mnist':
            # Load test data (Fashion MNIST)
            tf_bridge.keras_load_mnistfashion_test_data(test_data)

    # Initialize the model sequence
    if args.model_keras:
        # Create all other layers from the Keras Sequential model
        model_sequence.load_single(args.model_keras)
    elif args.sequence_keras:
        model_sequence.load_sequence(args.sequence_keras)

    network_view = MainNetworkView(app, model_sequence, test_data)

    # Prepare the layout
    def serve_layout():
        has_request = flask.has_request_context()

        layout = network_view.layout(has_request)
        return layout

    app.layout = serve_layout

    # Install the callbacks
    network_view.callbacks()

    # Run the server, will call the layout
    app.run_server(debug=args.debug)


def main():
    """ Application main entry point """
    args = parse_arguments()
    run_app(args)
