#
# Python application main entry point
#

from .main_model_selection import MainModelSelection
from .main_network_view import MainNetworkView
from .TestData import TestData
from .bridge import tensorflow_datasets as tf_bridge
from .bridge.KerasModelSequence import KerasModelSequence
from .widgets.Router import Router
from .widgets import font_awesome

import dash
import dash_bootstrap_components as dbc

import argparse
import logging


def parse_arguments():
    """ Parse command line arguments """

    # Command line options
    parser = argparse.ArgumentParser()
    parser_group = parser.add_mutually_exclusive_group(required=True)
    parser_group.add_argument("--model-directories", help="Comma separated list of directories to select models from")
    parser_group.add_argument("--model-keras", "-m", help="Load a Keras model from file")
    parser_group.add_argument("--sequence-keras", "-s",
                              help="Load sequence of Keras checkpoints following the pattern "
                                   "'dirpath/model_prefix{epoch}'")
    parser.add_argument("--test-dataset", "-t", help="Load a predefined test dataset (mnist, fashion-mnist, cifar-10)")
    parser.add_argument("--debug", help="Set Dash in debug mode", dest="debug", default=False, action='store_true')
    parser.add_argument("--sequence-pattern", default="{model}_{epoch}", help="Pattern to apply to detect sequences")
    parser.add_argument("--log-level", default="WARNING", help="Log level in (DEBUG, INFO, WARNING, ERROR)")
    parser.parse_args()

    # Handle command line arguments
    return parser.parse_args()


def run_app(args):
    """ Run the app """

    logger = logging.getLogger('dnnviewer')

    # create console handler
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    numeric_level = getattr(logging, args.log_level.upper(), None)
    if isinstance(numeric_level, int):
        logger.setLevel(numeric_level)
        console_handler.setLevel(numeric_level)

    # Create App, set stylesheets
    app = dash.Dash(__name__,
                    assets_folder="assets",
                    external_stylesheets=[dbc.themes.BOOTSTRAP, font_awesome.CDN_CSS_URL])
    app.title = 'DNN Viewer'

    # Test data
    test_data = TestData()
    if args.test_dataset:
        tf_bridge.keras_load_test_data(args.test_dataset, test_data)
        if not test_data.has_test_sample:
            logger.error('Unable to load dataset %s', args.test_dataset)

    # Model sequence : currently only supporting from Keras
    model_sequence = KerasModelSequence(test_data)

    # Initialize the model sequence
    if args.model_keras:
        # Create all other layers from the Keras Sequential model
        model_sequence.load_single(args.model_keras)
    elif args.sequence_keras:
        model_sequence.load_sequence(args.sequence_keras)

    # Top level pages
    router = Router()
    if model_sequence.number_epochs > 0:
        # Model already selected => single page on the main network view
        router.add_route('/', MainNetworkView(app, model_sequence, test_data, False))
    else:
        router.add_route('/', MainModelSelection(app, model_sequence, test_data,
                                                 args.model_directories.split(','), args.sequence_pattern))
        router.add_route('/network-view', MainNetworkView(app, model_sequence, test_data, True))

    def main_layout():
        return router.layout()

    # Top level layout : manage URL + current page content
    app.layout = main_layout

    # Install the callbacks
    [p.callbacks() for p in router.pages.values()]

    router.callbacks(app)

    # Run the server, will call the layout
    app.run_server(debug=args.debug)


def main():
    """ Application main entry point """
    args = parse_arguments()
    run_app(args)
