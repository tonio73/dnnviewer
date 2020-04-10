from . import model_sequence, grapher, AbstractDashboard
from .panes import top, center, bottom

import dash_bootstrap_components as dbc


class MainNetworkView(AbstractDashboard):

    """ Main dashboard with the network view in the middle """

    def __init__(self):
        self.panes = [top.TopPane(), center.CenterPane(), bottom.BottomPane()]

    def render(self, has_request):

        if has_request:
            # Force loading first model of sequence
            model_sequence.first_epoch(grapher)

            # Prepare rendering of panes
            [pane.render() for pane in self.panes]

    def layout(self, has_request):
        # Top app layout
        return dbc.Container([
            *[pane.get_layout() for pane in self.panes]
        ], fluid=True)

    def callbacks(self):

        [pane.callbacks() for pane in self.panes]