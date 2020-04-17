from . import AbstractDashboard, TestData
from .bridge import AbstractModelSequence, tensorflow_datasets as tf_ds_bridge
from .widgets import font_awesome

import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

import logging


class MainModelSelection(AbstractDashboard):
    """ Select model (sequence) and test data """

    def __init__(self, app, model_sequence: AbstractModelSequence, test_data: TestData.TestData,
                 model_directories, model_sequence_pattern):
        self.app = app
        self.model_sequence = model_sequence
        self.test_data = test_data
        self.model_directories = model_directories
        self.model_sequence_pattern = model_sequence_pattern

    def layout(self, has_request: bool):
        """ @return layout """

        return dbc.Container([html.H1([font_awesome.icon('binoculars'),
                                       html.Span('Deep Neural Network Viewer', style={'marginLeft': '15px'}),
                                       ]),
                              self._model_selection_form()
                              ])

    def callbacks(self):
        """ Setup callbacks """

        logger = logging.getLogger(__name__)

        @self.app.callback(Output('url-path', 'pathname'),
                           [Input('model-selection-submit', 'n_clicks')],
                           [State('model-selection-dropdown', 'value'),
                            State('test-data-selection-dropdown', 'value')])
        def url_redirect(n_clicks, model_path, test_dataset_id):
            if n_clicks is None:
                logger.warning('Prevent URL path to /network_view since n_clicks=%s', n_clicks)
                raise PreventUpdate

            if '{epoch}' in model_path:
                self.model_sequence.load_sequence(model_path)
            else:
                self.model_sequence.load_single(model_path)

            if self.model_sequence.number_epochs == 0:
                raise PreventUpdate

            self.test_data.reset()
            if test_dataset_id is not None:
                tf_ds_bridge.keras_load_test_data(test_dataset_id, self.test_data)
                if not self.test_data.has_test_sample:
                    logger.error('Unable to load dataset %s', test_dataset_id)

            return '/network-view'

        @self.app.callback(Output('model-selection-submit', 'disabled'),
                           [Input('model-selection-dropdown', 'value')])
        def model_validate(model_path):
            return model_path is None or len(model_path) == 0

        @self.app.callback(Output('model-selection-dropdown', 'options'),
                           [Input('model-selection-refresh', 'n_clicks')])
        def refresh_models(n_clicks):
            if n_clicks is None:
                raise PreventUpdate

            model_paths = self.model_sequence.list_models(self.model_directories, self.model_sequence_pattern)
            return [{'label': path, 'value': path} for path in model_paths]

    def _model_selection_form(self):

        model_paths = self.model_sequence.list_models(self.model_directories, self.model_sequence_pattern)
        test_datasets = tf_ds_bridge.keras_test_data_listing()

        return dbc.Form([
            # Model selection
            dbc.FormGroup([
                dbc.Label("Select a DNN model"),
                dcc.Dropdown(id='model-selection-dropdown',
                             value=None,
                             options=[{'label': path, 'value': path} for path in model_paths]
                             )
            ]),
            # Test data
            dbc.FormGroup([
                dbc.Label("Test data (optional)"),
                dcc.Dropdown(id='test-data-selection-dropdown',
                             value=None,
                             options=[{'label': test_datasets[i], 'value': i} for i in test_datasets]
                             )
            ]),
            # Submit
            dbc.Row([dbc.Col(dbc.Button("OK", id='model-selection-submit', color='primary', block=True), xs=2),
                     dbc.Col(dbc.Button(id='model-selection-refresh', children=font_awesome.icon('sync')))])
        ])
