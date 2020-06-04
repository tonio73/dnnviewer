from . import AbstractDashboard
from .TestData import TestData
from .Grapher import Grapher
from .Progress import Progress
from .bridge import AbstractModelSequence, tensorflow_datasets as tf_ds_bridge
from .bridge.AbstractModelSequence import ModelError
from .widgets import font_awesome

import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

import logging
# Using Threads until asyncio is supported by Flask
from threading import Thread

_logger = logging.getLogger(__name__)


class MainModelSelection(AbstractDashboard):
    """ Select model (sequence) and test data """

    supported_tasks = {'classification': "Classification (generic)",
                       'classification_image': "Image classification",
                       'misc': 'Other'}

    def __init__(self, app, model_selection, model_sequence: AbstractModelSequence, test_data: TestData,
                 grapher: Grapher):
        self.app = app
        self.model_selection = model_selection
        self.model_sequence = model_sequence
        self.test_data = test_data
        self.grapher = grapher
        self.progress = Progress()
        self.model_path = None

    def reset(self):

        self.progress.reset()

        # Preselected model path if any
        self.model_path = (self.model_selection['model'] or self.model_selection['sequence'])

        if self.model_path is not None:
            self._load_model(self.model_path, self.model_selection['test_dataset'])

    def layout(self, has_request: bool):
        """ @return layout """

        if has_request:
            self.reset()

        # Directly go to loading if pre-selection
        loading = True if self.model_path else False

        return dbc.Container([dcc.Interval(id='model-selection-loading-refresh', interval=500, disabled=not loading,
                                           max_intervals=100),
                              html.H1([font_awesome.icon('binoculars'),
                                       html.Span('Deep Neural Network Viewer',
                                                 style={'marginLeft': '15px'})]),
                              html.Div(id='model-selection-loading', hidden=not loading),
                              html.Div(id='model-selection-wrap', hidden=loading,
                                       children=self._model_selection_form())
                              ])

    def callbacks(self):
        """ Setup callbacks """
        @self.app.callback([Output('model-selection-loading-refresh', 'disabled'),
                            Output('model-selection-loading', 'hidden'),
                            Output('model-selection-wrap', 'hidden')],
                           [Input('model-selection-submit', 'n_clicks')],
                           [State('model-selection-dropdown', 'value'),
                            State('test-data-selection-dropdown', 'value')])
        def load_model(n_clicks, model_path, test_dataset_id):
            if n_clicks is None:
                _logger.warning('Prevent loading since n_clicks=%s', n_clicks)
                raise PreventUpdate

            self.model_path = model_path
            self._load_model(model_path, test_dataset_id)

            return False, False, True

        @self.app.callback(Output('url-path', 'pathname'),
                           [Input('model-selection-loading-refresh', 'n_intervals')])
        def loading_completed(n_intervals):
            if n_intervals is None:
                _logger.debug('Prevent URL redirect as refresh counter undefined')
                raise PreventUpdate

            if self.progress.num_steps is None:
                _logger.debug('Prevent URL redirect as progress setup incomplete')
                raise PreventUpdate

            status = self.progress.get_status()
            if status[0] < self.progress.num_steps:
                # Loading incomplete
                raise PreventUpdate

            if status[1] == Progress.ERROR:
                _logger.debug('Prevent URL redirect as loading completed with error')
                raise PreventUpdate

            return '/network-view'

        @self.app.callback(Output('model-selection-loading', 'children'),
                           [Input('model-selection-loading-refresh', 'n_intervals')])
        def model_loading_progress(n_intervals):
            if n_intervals is None:
                _logger.debug('Prevent loading status update')
                raise PreventUpdate

            if self.progress.num_steps is None:
                _logger.debug('Prevent loading status update as progress setup incomplete')
                raise PreventUpdate

            status = self.progress.get_status()
            progress_color = "danger" if status[1] == Progress.ERROR else "info"
            progress_message = 'Step %d - %s' % (self.progress.current_step, status[2])
            progress_percent = int(self.progress.current_step / self.progress.num_steps * 100)
            return [html.H2('Loading model "%s"' % self.model_path),
                    dbc.Progress(value=progress_percent, striped=True, color=progress_color),
                    html.H3(progress_message),
                    html.H3('On going: %s' % self.progress.next if self.progress.next else '')]

        @self.app.callback(Output('model-selection-submit', 'disabled'),
                           [Input('task-selection-dropdown', 'value'),
                            Input('model-selection-dropdown', 'value')])
        def model_validate(task, model_path):
            return task is None or len(task) == 0 \
                    or model_path is None or len(model_path) == 0

        @self.app.callback(Output('model-selection-dropdown', 'options'),
                           [Input('model-selection-refresh', 'n_clicks')])
        def refresh_models(n_clicks):
            if n_clicks is None:
                raise PreventUpdate

            model_paths = self.model_sequence.list_models(self.model_selection['directories'],
                                                          self.model_selection['pattern'])
            return [{'label': path, 'value': path} for path in model_paths]

    def _model_selection_form(self):

        model_paths = self.model_sequence.list_models(self.model_selection['directories'],
                                                      self.model_selection['pattern'])
        test_datasets = tf_ds_bridge.keras_test_data_listing()

        return dbc.Form([
            # Model selection
            dbc.FormGroup([
                dbc.Label("What is the task?"),
                dcc.Dropdown(id='task-selection-dropdown',
                             value=None,
                             options=[{'label': item[1], 'value': item[0]} for item in self.supported_tasks.items()]
                             )
            ]),
            # Model selection
            dbc.FormGroup([
                dbc.Label("Select a DNN model"),
                dcc.Dropdown(id='model-selection-dropdown',
                             value=self.model_selection['model'] or self.model_selection['sequence'],
                             options=[{'label': path, 'value': path} for path in model_paths]
                             )
            ]),
            # Test data
            dbc.FormGroup([
                dbc.Label("Test data (optional)"),
                dcc.Dropdown(id='test-data-selection-dropdown',
                             value=self.model_selection['test_dataset'],
                             options=[{'label': test_datasets[i], 'value': i} for i in test_datasets]
                             )
            ]),
            # Submit
            dbc.Row([dbc.Col(dbc.Button("OK", id='model-selection-submit', color='primary', block=True), xs=2),
                     dbc.Col(dbc.Button(id='model-selection-refresh', children=font_awesome.icon('sync')))])
        ])

    def _load_model(self, model_path, test_dataset_id):
        thread = Thread(target=load_model_and_data, args=(self, model_path, test_dataset_id))
        thread.daemon = True
        thread.start()


def load_model_and_data(self, model_path, test_dataset_id):
    """ Perform model and test data loading within thread """

    _logger.info("Start loading model '%s'", model_path)

    # Load sequence, test dataset, model #0
    num_steps = 3 if test_dataset_id is not None else 2
    self.progress.reset(num_steps)
    self.test_data.reset()

    if '{epoch}' in model_path:
        self.model_sequence.load_sequence(model_path)
    else:
        self.model_sequence.load_single(model_path)

    if self.model_sequence.number_epochs == 0:
        _logger.error('No model loaded in sequence from path: %s', model_path)
        self.progress.forward(1, Progress.ERROR, "Model sequence is empty")
        return

    self.progress.forward(1, Progress.INFO, "Model sequence initialized")

    if test_dataset_id is not None:
        self.progress.set_next("Loading test data (may take some time if download is necessary")
        tf_ds_bridge.keras_load_test_data(test_dataset_id, self.test_data)
        if not self.test_data.has_test_sample:
            _logger.error('Unable to load dataset %s', test_dataset_id)
            self.progress.forward(1, Progress.ERROR, 'Unable to load dataset %s' % test_dataset_id)
            return
        self.progress.forward(1, Progress.INFO, "Test dataset loaded")

    self.progress.set_next("Load model")

    try:
        # Force loading first model of sequence
        self.model_sequence.first_epoch(self.grapher)
    except ModelError as e:
        self.progress.forward(1, Progress.ERROR, "Error while loading model: %s" % e.message)
        return
    except Exception as e:
        self.progress.forward(1, Progress.ERROR, "Error while loading model: %s" % str(e))
        return

    _logger.info("Model loaded '%s'", model_path)
    self.progress.forward(1, Progress.INFO, "Model loaded")

    self.progress.set_next('Opening the main dashboard')
