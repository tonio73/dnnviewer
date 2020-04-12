from . import AbstractDashboard, TestData
from .bridge import AbstractModelSequence
from .Grapher import Grapher
from .panes import top, center, bottom

import dash_bootstrap_components as dbc


class MainNetworkView(AbstractDashboard):

    """ Main dashboard with three panes:
        - top title and toolbar
        - the network view in the middle
        - bottom sample data, layer and unit details, activation maps
    """

    def __init__(self, app, model_sequence: AbstractModelSequence, test_data: TestData.TestData):
        # Dash app
        self.app = app
        # DNN model as single or sequence over epochs
        self. model_sequence = model_sequence
        # Test data wrapper
        self.test_data = test_data
        # Graphical representation of DNN
        self.grapher: Grapher = Grapher()
        # Panes
        self.panes = [top.TopPane(), center.CenterPane(), bottom.BottomPane()]

    def layout(self, has_request):

        if has_request:
            # Force loading first model of sequence
            self.model_sequence.first_epoch(self.grapher)

            # Prepare rendering of panes
            [pane.render(self.grapher) for pane in self.panes]

        # Layout
        return dbc.Container([
            *[pane.get_layout(self.model_sequence, self.grapher, self.test_data) for pane in self.panes]
        ], fluid=True)

    def callbacks(self):

        [pane.callbacks(self.app, self.model_sequence, self.grapher, self.test_data) for pane in self.panes]
