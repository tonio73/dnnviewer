#
# Python application main entry point
#

from .main_model_selection import MainModelSelection
from .main_network_view import MainNetworkView
from .TestData import TestData
from .Grapher import Grapher
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

    # Parameters for model selection
    model_selection = {}

    # Test data
    test_data = TestData()
    model_selection['test_dataset'] = args.test_dataset

    # Model sequence : currently only supporting from Keras
    model_sequence = KerasModelSequence(test_data)

    # Initialize the model sequence
    model_selection['model'] = args.model_keras
    model_selection['sequence'] = args.sequence_keras
    preselection = True and model_selection['model'] or model_selection['sequence']

    # Model selection
    if args.model_directories:
        model_selection['directories'] = args.model_directories.split(',')
    else:
        model_selection['directories'] = []
    model_selection['pattern'] = args.sequence_pattern

    # Top level pages
    grapher = Grapher()
    router = Router()
    router.add_route('/', MainModelSelection(app, model_selection, model_sequence, test_data, grapher))
    router.add_route('/network-view', MainNetworkView(app, model_sequence, test_data, grapher, not preselection))

    def main_layout():
        return router.layout()

    # Top level layout : manage URL + current page content
    app.layout = main_layout

    # Install the callbacks
    [p.callbacks() for p in router.pages.values()]

    router.callbacks(app)

    # Run the server, will call the layout
    app.run_server(debug=args.debug)

    app.serve_routes()

def main():
    """ Application main entry point """
    args = parse_arguments()
    run_app(args)
