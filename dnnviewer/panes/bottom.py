#
# Bottom pane of the application: 4 quadrants with input sample, layer, unit and activation/saliency maps
#

from . import AbstractPane
from .. import app, grapher, test_data, model_sequence
from ..layers.AbstractLayer import AbstractLayer
from ..widgets import tabs
from ..imageutils import array_to_img_src

import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate


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

        return dbc.Row(style={'marginTop': '10px', 'marginBottom': '20px'}, children=[
            # Input sample selection
            dbc.Col(md=2, align='start',
                    children=html.Div(className='detail-section',
                                      children=[html.H5('Test sample'),
                                                dcc.Dropdown(
                                                    id='bottom-select-test-sample',
                                                    style={'marginTop': '12px'},
                                                    value=self.test_sample_init,
                                                    options=[{'label': "%d (%s)" % (i, test_data.output_classes[c]),
                                                              'value': i}
                                                             for i, c in
                                                             enumerate(test_data.y[:self.max_test_samples])]
                                                ),
                                                html.P([html.Img(id='bottom-test-sample-img', alt='Sample input')],
                                                       className='thumbnail', hidden=not test_data.has_test_sample)
                                                ])),

            # Layer information
            dbc.Col(md=3, align='start',
                    children=[
                        dcc.Store(id='bottom-layer-click-data'),
                        html.Div(className='detail-section',
                                 children=[html.Div(id='bottom-layer-title'),
                                           html.Div(id='bottom-layer-tabs',
                                                    children=dummy_layer.get_layer_tabs(None))
                                           ])
                    ]),

            # Unit information
            dbc.Col(md=4, align='start',
                    children=html.Div(className='detail-section',
                                      children=[html.Div(id='bottom-unit-title', children=html.H5('Maps')),
                                                html.Div(id='bottom-unit-tabs',
                                                         children=dummy_layer.get_unit_tabs(0, None))
                                                ])),

            #  Maps
            dbc.Col(md=3, align='start',
                    children=html.Div(className='detail-section',
                                      children=[html.Div(id='bottom-maps-title',
                                                         children=html.H5('Maps')),
                                                html.Div(id='bottom-maps-tabs',
                                                         children=self._get_maps_tabs(None))]))
        ])

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

        @app.callback(Output('bottom-maps-tab-content', 'children'),
                      [Input("bottom-maps-tab-bar", "active_tab"),
                       Input('bottom-select-test-sample', 'value'),
                       Input('center-selected-unit', 'data'),
                       Input('top-epoch-index', 'data')])
        def update_activation_map(active_tab, index, selected_unit, _):
            if active_tab is not None \
                    and index is not None and test_data.x is not None \
                    and model_sequence \
                    and selected_unit:
                if active_tab == 'activation':
                    layer = grapher.layers[selected_unit['layer_idx']]
                    return layer.get_activation_map(model_sequence, test_data.x[index], selected_unit['unit_idx'])
            return []

        @app.callback(Output("bottom-layer-tab-content", "children"),
                      [Input("bottom-layer-tab-bar", "active_tab"),
                       Input('center-selected-unit', 'data')])
        def render_layer_tab_content(active_tab, selected_unit):
            """ layer info tab selected => update content """
            if active_tab and selected_unit is not None:
                layer = grapher.layers[selected_unit['layer_idx']]
                if layer is not None:
                    return layer.get_layer_tab_content(active_tab)
            return html.Div()

        @app.callback(Output("bottom-unit-tab-content", "children"),
                      [Input("bottom-unit-tab-bar", "active_tab")],
                      [State('center-selected-unit', 'data')])
        def render_unit_tab_content(active_tab, selected_unit):
            """ layer info tab selected => update content """
            if active_tab and selected_unit is not None:
                layer = grapher.layers[selected_unit['layer_idx']]
                if layer is not None:
                    return layer.get_unit_tab_content(selected_unit['unit_idx'], active_tab)
            return html.Div()

        @app.callback([Output('bottom-layer-title', 'children'), Output('bottom-layer-tabs', 'children'),
                       Output('bottom-unit-title', 'children'), Output('bottom-unit-tabs', 'children')],
                      [Input('center-selected-unit', 'data'), Input('top-epoch-index', 'data')],
                      [State('bottom-layer-tab-bar', 'active_tab'),
                       State('bottom-unit-tab-bar', 'active_tab')])
        def update_layer_info(selected_unit, _, active_layer_tab, active_unit_tab):
            """ Display selected layer/unit title and tabs """
            if selected_unit:
                layer = grapher.layers[selected_unit['layer_idx']]
                return layer.get_layer_title(), \
                       layer.get_layer_tabs(active_layer_tab), \
                       layer.get_unit_title(selected_unit['unit_idx']), \
                       layer.get_unit_tabs(selected_unit['unit_idx'], active_unit_tab)
            dummy_layer = AbstractLayer('dummy')
            return [], dummy_layer.get_layer_tabs(active_layer_tab), [], dummy_layer.get_unit_tabs(0, active_unit_tab)

        @app.callback(Output('center-main-view', 'clickData'),
                      [Input('bottom-layer-click-data', 'data')],
                      [State('center-main-view', 'clickData')])
        def update_unit_selection(click_data, network_click_data):
            """ Click on the layer figure => update the main selection's unit """
            if click_data and network_click_data:
                network_click_data['points'][0]['pointNumber'] = click_data['points'][0]['pointNumber']
            return network_click_data

        @app.callback(Output('bottom-layer-click-data', 'data'),
                      [Input('bottom-layer-figure', 'clickData')])
        def update_layer_click_data(click_data):
            if click_data:
                return click_data
            raise PreventUpdate

        return

    def _get_maps_tabs(self, active_tab):
        """ Tabs for the maps """
        return tabs.make('bottom-maps', {'activation': 'Activation'}, active_tab)
