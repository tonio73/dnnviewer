#
# Top pane of the application
#

from .. import app
from ..widgets import font_awesome

import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State


def render():
    return


def get_layout():
    """ Get pane layout """
    return dbc.Row([
                   dbc.Col([html.H1([font_awesome.icon('binoculars'), html.Span('Deep Neural Network Viewer',
                                                                                style={'marginLeft': '15px'})])
                            ])
               ])

def callbacks():
    """ Local callbacks """
    return