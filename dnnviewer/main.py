#!/usr/bin/env python3

#
# Python application main entry point
#

from .main_model_selection import MainModelSelection
from .main_network_view import MainNetworkView
from .TestData import TestData
from .bridge import tensorflow as tf_bridge
from .bridge.KerasModelSequence import KerasModelSequence
from .widgets import font_awesome

import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import flask

import argparse
import logging


def parse_arguments():
    """ Parse command line arguments """

    # Command line options
    parser = argparse.ArgumentParser()
    parser_group = parser.add_mutually_exclusive_group(required=True)
    parser_group.add_argument("--model-keras", "-m", help="Load a Keras model from file")
    parser_group.add_argument("--sequence-keras", "-s",
                              help="Load sequence of Keras checkpoints following the pattern "
                                   "'dirpath/model_prefix{epoch}'")
    parser_group.add_argument("--model-directory", help="Select a directory to load models from ")
    parser.add_argument("--test-dataset", "-t", help="Load a predefined test dataset (mnist, fashion-mnist, cifar-10)")
    parser.add_argument("--debug", help="Set Dash in debug mode", dest="debug", default=False, action='store_true')
    parser.parse_args()

    # Handle command line arguments
    return parser.parse_args()


def run_app(args):
    """ Run the app """

    logger = logging.getLogger(__name__)

    # Create App, set stylesheets
    app = dash.Dash(__name__,
                    assets_folder="assets",
                    external_stylesheets=[dbc.themes.BOOTSTRAP, font_awesome.CDN_CSS_URL])
    app.title = 'DNN Viewer'

    # Test data
    if args.test_dataset:
        test_data = tf_bridge.keras_load_test_data(args.test_dataset)
        if test_data is None:
            logger.error('Unable to load dataset %s', args.test_dataset)
            test_data = TestData()
    else:
        test_data = TestData()

    # Model sequence : currently only supporting from Keras
    model_sequence = KerasModelSequence(test_data)

    # Initialize the model sequence
    if args.model_keras:
        # Create all other layers from the Keras Sequential model
        model_sequence.load_single(args.model_keras)
    elif args.sequence_keras:
        model_sequence.load_sequence(args.sequence_keras)

    # Top level pages
    pages = {
        '/': MainModelSelection(app, model_sequence, test_data),
        '/network-view': MainNetworkView(app, model_sequence, test_data)
    }

    def main_layout():
        if flask.has_request_context():
            return html.Div([dcc.Location(id='url-path', refresh=False),
                             dcc.Store(id='saved-url-path'),
                             html.Div(id='page-content', children=html.Div('Request coming...'))])
        else:
            return html.Div([dcc.Location(id='url-path', refresh=False),
                             dcc.Store(id='saved-url-path'),
                             html.Div(id='page-content',
                                      # All pages for validation
                                      children=[page.layout(False) for page in pages.values()])])

    # Top level layout : manage URL + current page content
    app.layout = main_layout

    # Install the callbacks
    [p.callbacks() for p in pages.values()]

    # Dispatch URL path name
    @app.callback([Output('page-content', 'children'),
                   Output('saved-url-path', 'data')],
                  [Input('url-path', 'pathname')],
                  [State('saved-url-path', 'data')])
    def update_layout(path_name, saved_path):
        has_request = flask.has_request_context()

        logger.info("Reaching path %s from %s", path_name, saved_path)
        if path_name == saved_path:
            raise PreventUpdate

        if has_request and path_name in pages:
            return pages[path_name].layout(True), path_name
        return '404', path_name

    # Run the server, will call the layout
    app.run_server(debug=args.debug)


def main():
    """ Application main entry point """
    args = parse_arguments()
    run_app(args)
