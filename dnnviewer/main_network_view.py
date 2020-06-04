from . import AbstractDashboard, TestData
from .bridge import AbstractModelSequence
from .Grapher import Grapher
from .panes import top, center, bottom
from .widgets import font_awesome

import dash_bootstrap_components as dbc
import dash_html_components as html
import dash_core_components as dcc


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
        self.enable_navigation = enable_navigation

    def layout(self, has_request):
        if has_request:
            # Prepare rendering of panes
            [pane.render(self.grapher) for pane in self.panes]

            # Layout
            if self.model_sequence.number_epochs == 0:
                if self.enable_navigation:
                    style = {'margin-left': '15px'}
                    return_selection = dcc.Link(html.H3([font_awesome.icon('arrow-left'),
                                                         html.Span('Back to model selection', style=style)]), href='/')
                else:
                    return_selection = html.P()

                return dbc.Container([html.H1([font_awesome.icon('binoculars'),
                                               html.Span('Deep Neural Network Viewer',
                                                         style={'marginLeft': '15px'})]),
                                      html.H2('No model loaded'),
                                      return_selection
                                      ])
            else:
                return dbc.Container([
                    *[pane.get_layout(self.model_sequence, self.grapher, self.test_data) for pane in self.panes]
                ], fluid=True)
        else:
            return dbc.Container([
                *[pane.get_layout(self.model_sequence, self.grapher, self.test_data) for pane in self.panes]
            ], fluid=True)

    def callbacks(self):

        [pane.callbacks(self.app, self.model_sequence, self.grapher, self.test_data) for pane in self.panes]
