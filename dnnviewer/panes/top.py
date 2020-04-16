from . import AbstractPane
from ..widgets import font_awesome

from dash import callback_context
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output

import logging


class TopPane(AbstractPane):
    """ Top pane of the application """

    def __init__(self, enable_navigation):
        self.enable_navigation = enable_navigation

    def get_layout(self, model_sequence, grapher, test_data):
        """ Get pane layout """
        return dbc.Row([
            dcc.Store('top-epoch-index', data=model_sequence.current_epoch_index, storage_type='session'),
            dbc.Col([html.H1([html.Span(dcc.Link(font_awesome.icon('arrow-left'), href='/'),
                                        hidden=not self.enable_navigation,
                                        style={'marginRight': '15px'}),
                              html.Span('Deep Neural Network Viewer')])
                     ], md=9),
            dbc.Col(html.Div([dbc.Button(font_awesome.icon('fast-backward'), className="mr-1",
                                         id='top-fast-backward'),
                              dbc.Button(font_awesome.icon('step-backward'), className="ml-1 mr-1",
                                         id='top-step-backward'),
                              html.Span('Epoch %d / %d' %
                                        (model_sequence.current_epoch_index + 1, model_sequence.number_epochs),
                                        className="ml-1 mr-1", id='top-epoch-display'),
                              dbc.Button(font_awesome.icon('step-forward'), className="ml-1 mr-1",
                                         id='top-step-forward'),
                              dbc.Button(font_awesome.icon('fast-forward'), className="ml-1",
                                         id='top-fast-forward'),
                              ], hidden=(model_sequence.number_epochs <= 1)),
                    md=3, align='center')
        ])

    def callbacks(self, app, model_sequence, grapher, test_data):
        """ Local callbacks """

        logger = logging.getLogger(__name__)

        @app.callback(Output('top-epoch-index', 'data'),
                      [Input('top-step-backward', 'n_clicks'),
                       Input('top-fast-forward', 'n_clicks'),
                       Input('top-step-forward', 'n_clicks'),
                       # There is normally no event trigger on load BUT it seems that there is one, the latest
                       Input('top-fast-backward', 'n_clicks')])
        def sequence_move(_0, _1, _2, _3):

            ctx = callback_context

            if not ctx.triggered:
                button_id = 'no_clicks'
            else:
                button_id = ctx.triggered[0]['prop_id'].split('.')[0]

            logger.debug('Triggering %s, %s, %s, %s, %s', button_id, _0, _1, _2, _3)

            if button_id == 'top-fast-backward':
                return model_sequence.first_epoch(grapher)
            elif button_id == 'top-step-backward':
                return model_sequence.previous_epoch(grapher)
            elif button_id == 'top-fast-forward':
                return model_sequence.last_epoch(grapher)
            elif button_id == 'top-step-forward':
                return model_sequence.next_epoch(grapher)

        @app.callback(Output('top-epoch-display', 'children'),
                      [Input('top-epoch-index', 'data')])
        def epoch_display(_):
            """ Set the text of the epoch display """
            return 'Epoch %d / %d' % (model_sequence.current_epoch_index + 1, model_sequence.number_epochs)

        return
