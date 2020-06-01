import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from dash.exceptions import CallbackException
import flask

import logging

_logger = logging.getLogger(__name__)


class Router:
    """ Pseudo widget to manage routing to pages """

    def __init__(self):
        self.pages = {}
        self.previous_path = ''

    def add_route(self, path, page):
        self.pages[path] = page

    def layout(self):
        if flask.has_request_context():
            return html.Div([dcc.Location(id='url-path', refresh=False),
                             html.Div(id='page-content', children=html.Div('Request coming...'))])
        else:
            return html.Div([dcc.Location(id='url-path', refresh=False),
                             html.Div(id='page-content',
                                      # All pages for validation
                                      children=[page.layout(False) for page in self.pages.values()])])

    def callbacks(self, app):

        @app.callback(Output('page-content', 'children'),
                      [Input('url-path', 'pathname')])
        def dispatch(path_name):
            has_request = flask.has_request_context()

            # Work around double firing of route : https://github.com/plotly/dash/issues/1049
            _logger.info("Reaching path: %s, previous: %s", path_name, self.previous_path)
            self.previous_path = path_name

            if has_request and path_name in self.pages:
                return self.pages[path_name].layout(True)
            return '404'
