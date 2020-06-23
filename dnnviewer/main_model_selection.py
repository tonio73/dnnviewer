from . import AbstractDashboard
from dnnviewer.dataset.DataSet import DataSet
from dnnviewer.dataset import generators
from .Grapher import Grapher
from .Progress import Progress
from .bridge import DatasetError, tensorflow_datasets as tf_ds_bridge
from .bridge.AbstractModelSequence import AbstractModelSequence
from .bridge import ModelError
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

    test_dataset_modes = {DataSet.MODE_UNKNOWN: "None",
                          DataSet.MODE_FILESET: "Tensorflow dataset",
                          DataSet.MODE_GENERATOR: "Generator"}

    supported_tasks = {'classification': "Classification (generic)",
                       'classification_image': "Image classification",
                       'misc': 'Other'}

    available_generators = {generators.RANDOM_NORMAL: 'Random normal (Gauss with mean 0 and variance 1.0)'}

    def __init__(self, app, model_config, model_sequence: AbstractModelSequence, test_data: DataSet,
                 grapher: Grapher):
        self.app = app
        self.model_config = model_config
        self.model_sequence = model_sequence
        self.test_data: DataSet = test_data
        self.grapher = grapher
        self.progress = Progress()
        self.model_path = None

    def reset(self):

        self.progress.reset()

        # Preselected model path if any
        self.model_path = (self.model_config['model'] or self.model_config['sequence'])

        if self.model_path is not None:
            self._load_model(self.model_path, self.model_config['test_mode'], self.model_config['test_dataset'])

    def layout(self, has_request: bool):
        """ @return layout """

        if has_request:
            self.reset()

        # Directly go to loading if pre-selection
        loading = True if self.model_path else False

        return dbc.Container([dcc.Interval(id='model-selection-loading-refresh', interval=500, disabled=not loading,
                                           max_intervals=100),
                              html.H1([html.Span(font_awesome.icon('binoculars'), className='title-icon'),
                                       html.Span('Deep Neural Network Viewer')],
                                      className='title'),
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
                           [State('test-data-mode', 'value'),
                            State('model-selection-dropdown', 'value'),
                            State('test-data-selection-dropdown', 'value'),
                            State('test-generator-selection-dropdown', 'value')])
        def load_model(n_clicks, test_mode, model_path, test_dataset_id, generator_id):
            if n_clicks is None:
                _logger.warning('Prevent loading since n_clicks=%s', n_clicks)
                raise PreventUpdate

            self.model_path = model_path
            self._load_model(test_mode, model_path, test_dataset_id, generator_id)

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
            if status[1] == Progress.ERROR:
                progress_color = "danger"
            else:
                progress_color = "info"
            progress_message = 'Step %d - %s' % (self.progress.current_step, status[2])
            progress_percent = int(self.progress.current_step / self.progress.num_steps * 100)
            return [html.H2('Loading model "%s"' % self.model_path),
                    dbc.Progress(value=progress_percent, striped=True, color=progress_color),
                    html.H3(progress_message),
                    html.H3('On going: %s' % self.progress.next if self.progress.next else '')]

        @self.app.callback(Output('model-selection-submit', 'disabled'),
                           [Input('model-selection-dropdown', 'value')])
        def model_validate(model_path):
            return model_path is None or len(model_path) == 0

        @self.app.callback(Output('model-selection-dropdown', 'options'),
                           [Input('model-selection-refresh', 'n_clicks')])
        def refresh_models(n_clicks):
            if n_clicks is None:
                raise PreventUpdate

            model_paths = self.model_sequence.list_models(self.model_config['directories'],
                                                          self.model_config['pattern'])
            return [{'label': path, 'value': path} for path in model_paths]

        @self.app.callback([Output('test-data-generator', 'hidden'),
                            Output('test-data-tensorflow-dataset', 'hidden')],
                           [Input('test-data-mode', 'value')])
        def test_data_mode(mode):
            return mode is not DataSet.MODE_GENERATOR, mode is not DataSet.MODE_FILESET

        return  # Set callbacks

    def _model_selection_form(self):

        model_paths = self.model_sequence.list_models(self.model_config['directories'],
                                                      self.model_config['pattern'])
        test_datasets = tf_ds_bridge.list_test_data()

        return dbc.Form([
            # Model
            dbc.FormGroup(html.H3("Model")),
            dbc.FormGroup([
                dbc.Label("Select a DNN model"),
                dcc.Dropdown(id='model-selection-dropdown',
                             value=self.model_config['model'] or self.model_config['sequence'],
                             options=[{'label': path, 'value': path} for path in model_paths]
                             )
            ]),
            # Test data
            dbc.FormGroup(html.H3("Test data (optional)")),
            dbc.FormGroup([
                dbc.Label("Select data type"),
                dbc.RadioItems(id='test-data-mode',
                               value=self.model_config['test_mode'],
                               options=[{'label': label, 'value': key} \
                                        for (key, label) in self.test_dataset_modes.items()],
                               inline=True
                               )
            ]),
            # Generator as dataset
            html.Div(id='test-data-generator',
                     children=dbc.FormGroup([
                         dbc.Label("Input generator"),
                         dcc.Dropdown(id='test-generator-selection-dropdown',
                                      value=self.model_config['test_dataset'],
                                      options=[{'label': caption, 'value': key}  \
                                               for key, caption in self.available_generators.items()]
                                      ),
                     ])),
            # Tensorflow dataset
            html.Div(id='test-data-tensorflow-dataset',
                     children=dbc.FormGroup([
                         dbc.Label("Tensorflow Dataset"),
                         dcc.Dropdown(id='test-data-selection-dropdown',
                                      value=self.model_config['test_dataset'],
                                      options=[{'label': ds, 'value': ds} for ds in test_datasets]
                                      )
                     ], inline=True)),
            # Submit
            dbc.Row([dbc.Col(dbc.Button("OK", id='model-selection-submit', color='primary', block=True),
                             xs=2, className=''),
                     dbc.Col(dbc.Button(id='model-selection-refresh', children=font_awesome.icon('sync')))])
        ])

    def _load_model(self, test_mode, model_path, test_dataset_id, generator_id=None):
        thread = Thread(target=load_model_and_data, args=(self, test_mode, model_path, test_dataset_id, generator_id))
        thread.daemon = True
        thread.start()


def load_model_and_data(self, test_mode, model_path, test_dataset_id, generator_id):
    """ Perform model and test data loading within thread """

    _logger.info("Start loading model '%s'", model_path)

    # Load sequence, test dataset, model #0
    if self.model_config['test_mode'] is DataSet.MODE_FILESET and test_dataset_id is not None:
        num_steps = 4
    else:
        num_steps = 2
    self.progress.reset(num_steps)
    self.test_data.reset()

    # Initialize model sequence (even if single model in sequence)
    if '{epoch}' in model_path:
        self.model_sequence.load_sequence(model_path)
    else:
        self.model_sequence.load_single(model_path)

    if self.model_sequence.number_epochs == 0:
        _logger.error('No model loaded in sequence from path: %s', model_path)
        self.progress.forward(1, Progress.ERROR, "Model sequence is empty")
        return

    self.progress.forward(1, Progress.INFO, "Model sequence initialized")

    # Initialize test dataset if any
    if test_mode is DataSet.MODE_FILESET and test_dataset_id is not None:
        # Load test dataset
        self.progress.set_next("Loading test data (it takes up to few minutes at first attempt while downloading data)")
        try:
            tf_ds_bridge.load_test_data(test_dataset_id, self.test_data)
        except DatasetError as e:
            _logger.error('Unable to load dataset %s', test_dataset_id)
            self.progress.forward(1, Progress.ERROR, 'Unable to load dataset %s: %s' % (test_dataset_id, e.message))
            return

        if self.test_data.mode is not DataSet.MODE_FILESET:
            _logger.error('Unable to load dataset %s', test_dataset_id)
            self.progress.forward(1, Progress.ERROR, 'Unable to load dataset %s' % test_dataset_id)
            return
        self.progress.forward(1, Progress.INFO, "Test dataset loaded")

        # Prepare data
        self.progress.set_next('Format test data')
        try:
            self.model_sequence.format_test_data()
        except ModelError as e:
            self.progress.forward(1, Progress.ERROR, e.message)
            return

        self.progress.forward(1, Progress.INFO, "Test data formatted")

    if test_mode is DataSet.MODE_GENERATOR:
        if generator_id is not None:
            # For the time being, only supported generator is random normal
            self.model_sequence.setup_generator(generators.get_generators(generator_id))

    # Force loading first model of sequence
    self.progress.set_next("Load model")
    try:
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
