#
# Application wrapper containing the singletons
#

# from .bridge.AbstractModelSequence import AbstractModelSequence
from .bridge.KerasModelSequence import KerasModelSequence
from .Grapher import Grapher
from .TestData import TestData
from .widgets import font_awesome

import dash
import dash_bootstrap_components as dbc

# Graphical representation of DNN
grapher: Grapher = Grapher()

# Create App, set stylesheets
app = dash.Dash(__name__,
                assets_folder="assets",
                external_stylesheets=[dbc.themes.BOOTSTRAP, font_awesome.CDN_CSS_URL])
app.title = 'DNN Viewer'

# Manager for test data
test_data: TestData = TestData()

# Todo correct init without dependency on KerasModelSequence, using AbstractModelSequence
model_sequence: KerasModelSequence = KerasModelSequence(test_data)
