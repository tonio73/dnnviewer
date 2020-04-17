from . import AbstractDashboard, TestData
from .bridge import AbstractModelSequence
from .Grapher import Grapher
from .panes import top, center, bottom

import dash_bootstrap_components as dbc
import dash_html_components as html

import time


class MainNetworkView(AbstractDashboard):

    """ Main dashboard with three panes:
        - top title and toolbar
        - the network view in the middle
        - bottom sample data, layer and unit details, activation maps
    """

    def __init__(self, app, model_sequence: AbstractModelSequence, test_data: TestData.TestData, enable_navigation):
        # Dash app
        self.app = app
        # DNN model as single or sequence over epochs
        self. model_sequence = model_sequence
        # Test data wrapper
        self.test_data = test_data
        # Graphical representation of DNN
        self.grapher: Grapher = Grapher()
        # Panes
        self.panes = [top.TopPane(enable_navigation), center.CenterPane(), bottom.BottomPane()]

        self.refresh_count = 0

    def layout(self, has_request):

        # Work around double firing of route : https://github.com/plotly/dash/issues/1049
        self.refresh_count += 1

        if (self.refresh_count & 1) or not has_request:

            if has_request:
                # Force loading first model of sequence
                self.model_sequence.first_epoch(self.grapher)

                # Prepare rendering of panes
                [pane.render(self.grapher) for pane in self.panes]

            # Layout
            content = dbc.Container([
                *[pane.get_layout(self.model_sequence, self.grapher, self.test_data) for pane in self.panes]
            ], fluid=True)

            return content

        else:
            time.sleep(10.0)
            return html.Div(html.H2('There has been an issue during model loading, please reload this page'))

    def callbacks(self):

        [pane.callbacks(self.app, self.model_sequence, self.grapher, self.test_data) for pane in self.panes]
