#
# Application wrapper containing the singletons
#

from dnnviewerapp.Grapher import Grapher
from dnnviewerapp.TestData import TestData
from dnnviewerapp.widgets import font_awesome

import dash
import dash_bootstrap_components as dbc

# Graphical representation of DNN
grapher = Grapher()

# Create App, set stylesheets
app = dash.Dash(__name__,
                assets_folder="assets",
                external_stylesheets=[dbc.themes.BOOTSTRAP, font_awesome.CDN_CSS_URL])
app.title = 'DNN Viewer'


test_data = TestData()