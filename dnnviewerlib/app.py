#
# Application wrapper containing the singletons
#

from dnnviewerlib.Grapher import Grapher
from dnnviewerlib.TestData import TestData
from dnnviewerlib.widgets import font_awesome

import dash
import plotly.graph_objects as go
import dash_bootstrap_components as dbc

# Graphical representation of DNN
grapher = Grapher()

# Create App, set stylesheets
app = dash.Dash(__name__,
                assets_folder="assets",
                external_stylesheets=[dbc.themes.BOOTSTRAP, font_awesome.CDN_CSS_URL])
app.title = 'DNN Viewer'

# Main network view
main_view = go.Figure()
main_view.update_layout(margin=dict(l=10, r=10, b=30, t=30))


test_data = TestData()
