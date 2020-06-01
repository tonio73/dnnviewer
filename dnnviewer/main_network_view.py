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

    def __init__(self, app, model_sequence: AbstractModelSequence, test_data: TestData.TestData, grapher: Grapher,
                 enable_navigation):
        # Dash app
        self.app = app
        # DNN model as single or sequence over epochs
        self. model_sequence = model_sequence
        # Test data wrapper
        self.test_data = test_data
        # Graphical representation of DNN
        self.grapher = grapher
        # Panes
        self.panes = [top.TopPane(enable_navigation), center.CenterPane(), bottom.BottomPane()]

    def layout(self, has_request):
        if has_request:

            # Prepare rendering of panes
            [pane.render(self.grapher) for pane in self.panes]

        # Layout
        content = dbc.Container([
            *[pane.get_layout(self.model_sequence, self.grapher, self.test_data) for pane in self.panes]
        ], fluid=True)

        return content

    def callbacks(self):

        [pane.callbacks(self.app, self.model_sequence, self.grapher, self.test_data) for pane in self.panes]
