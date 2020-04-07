#
# Bottom pane of the application: 4 quadrants with input sample, layer, unit and activation/saliency maps
#

from . import AbstractPane
from .. import app, grapher, test_data, model_sequence
from ..layers.AbstractLayer import AbstractLayer

import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State

from ..imageutils import array_to_img_src
from ..widgets import activation_map


class BottomPane(AbstractPane):
    # Test sample to show on init
    test_sample_init = 0

    # maximum number of test sample to display in selectors
    max_test_samples = 40

    def render(self):
        return

    def get_layout(self):
        """ Get pane layout """

        dummy_layer = AbstractLayer('dummy')

        return dbc.Row([
            # Input sample selection
            dbc.Col([
                html.Div([html.Label('Select test sample'),
                          dcc.Dropdown(
                              id='bottom-select-test-sample',
                              value=self.test_sample_init,
                              options=[{'label': "%d (%s)" % (i, test_data.output_classes[c]), 'value': i} for i, c in
                                       enumerate(test_data.y[:self.max_test_samples])]
                          )]),
                html.P([html.Img(id='bottom-test-sample-img', alt='Sample input')],
                       className='thumbnail', hidden=not test_data.has_test_sample)
            ], md=2, align='start'),

            # Layer information
            dbc.Col([
                html.Label('Selected layer'),
                html.Div(className='detail-section',
                         children=[
                             html.Div(id='bottom-layer-title'),
                             html.Div(id='bottom-layer-tabs',
                                      children=dummy_layer.get_layer_tabs(None))
                         ])
            ], md=3, align='start'),

            # Unit information
            dbc.Col([
                html.Label('Selected unit'),
                html.Div(className='detail-section',
                         children=[
                             html.Div(id='bottom-unit-title'),
                             html.Div(id='bottom-unit-tabs',
                                      children=dummy_layer.get_unit_tabs(0, None))
                         ])
            ], md=4, align='start'),

            # Activation maps
            dbc.Col([html.Label('Activation maps'),
                     html.Div(id='bottom-activation-maps')],
                    md=3, align='start')
        ], style={'marginTop': '10px', 'marginBottom': '20px'})

    def callbacks(self):
        """ Local callbacks """

        @app.callback(Output('bottom-test-sample-img', 'src'),
                      [Input('bottom-select-test-sample', 'value')])
        def update_test_sample(index):
            """ Update the display of the selected test sample upon selection
                @return the image to be displayed as base64 encoded png
            """
            if index is not None and test_data.x is not None:
                img = test_data.x[index]
                return array_to_img_src(img)
            return ''

        @app.callback(Output('bottom-activation-maps', 'children'),
                      [Input('bottom-select-test-sample', 'value'),
                       Input('center-main-view', 'clickData'),
                       Input('top-epoch-index', 'data')])
        def update_activation_map(index, click_data, _):
            if index is not None and test_data.x is not None \
                    and model_sequence \
                    and click_data:
                layer, unit_idx = grapher.get_layer_unit_from_click_data(click_data)
                return activation_map.widget(model_sequence, layer, unit_idx, test_data.x[index])
            return []

        @app.callback(Output("bottom-layer-tab-content", "children"),
                      [Input("bottom-layer-tab-bar", "active_tab")],
                      [State('center-main-view', 'clickData')])
        def render_layer_tab_content(active_tab, network_click_data):
            """ layer info tab selected => update content """
            if active_tab and network_click_data is not None:
                layer, unit_idx = grapher.get_layer_unit_from_click_data(network_click_data)
                if layer is not None:
                    return layer.get_layer_tab_content(active_tab)
            return html.Div()

        @app.callback(Output("bottom-unit-tab-content", "children"),
                      [Input("bottom-unit-tab-bar", "active_tab")],
                      [State('center-main-view', 'clickData')])
        def render_unit_tab_content(active_tab, network_click_data):
            """ layer info tab selected => update content """
            if active_tab and network_click_data is not None:
                layer, unit_idx = grapher.get_layer_unit_from_click_data(network_click_data)
                if layer is not None:
                    return layer.get_unit_tab_content(unit_idx, active_tab)
            return html.Div()

        return
