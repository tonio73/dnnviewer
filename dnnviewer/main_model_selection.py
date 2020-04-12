from . import AbstractDashboard, TestData
from .bridge import AbstractModelSequence

import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output


class MainModelSelection(AbstractDashboard):
    """ Select model (sequence) and test data """

    def __init__(self, app, model_sequence: AbstractModelSequence, test_data: TestData.TestData):
        self.app = app
        self.model_sequence = model_sequence
        self.test_data = test_data

    def layout(self, has_request: bool):
        """ @return layout """

        if has_request:
            if self.model_sequence.number_epochs > 0:
                return dbc.Container([
                    dcc.Interval(id='model-selection-refresh', interval=1000, n_intervals=1),
                    html.H1('Model loading...')
                ])
            else:
                return dbc.Container([html.H1('Error, no model loaded')])
        else:
            # Return all children for validation
            return dbc.Container([
                dcc.Interval(id='model-selection-refresh', interval=10000000, n_intervals=0)
            ])

    def callbacks(self):
        """ Setup callbacks """

        @self.app.callback(Output('url-path', 'pathname'),
                           [Input('model-selection-refresh', 'n_intervals')])
        def url_redirect(_):
            return '/network-view'

        return
