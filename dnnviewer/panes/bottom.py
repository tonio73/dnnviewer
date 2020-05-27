#
# Bottom pane of the application: 4 quadrants with input sample, layer, unit and activation/saliency maps
#

from . import AbstractPane
from ..layers.AbstractLayer import AbstractLayer
from ..widgets import tabs
from ..imageutils import array_to_img_src
from ..TestData import TestData

import plotly.graph_objects as go
from dash import callback_context
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State


class BottomPane(AbstractPane):
    # Test sample to show on init
    test_sample_init = 0

    # maximum number of test sample to display in selectors
    max_test_samples = 40

    def get_layout(self, model_sequence, grapher, test_data: TestData):
        """ Get pane layout """

        dummy_layer = AbstractLayer('dummy')

        if test_data.has_test_sample:
            test_data_selector = [html.H5('Test sample', key='bottom-test-data-title-text'),
                                  dcc.Dropdown(
                                      id='bottom-select-test-sample',
                                      style={'marginTop': '12px'},
                                      value=self.test_sample_init,
                                      options=[{'label': "%d (%s)" % (i, test_data.output_classes[c]),
                                                'value': i}
                                               for i, c in
                                               enumerate(test_data.y[:self.max_test_samples])]
                                  ),
                                  html.P(key='bottom-test-data-img-wrap',
                                         children=[
                                             html.Img(id='bottom-test-sample-img', key='bottom-test-data-img',
                                                      alt='Sample input')],
                                         className='thumbnail', hidden=not test_data.has_test_sample)
                                  ]
        else:
            test_data_selector = [html.H5('Test sample', key='bottom-test-data-title-text'),
                                  html.P('No test data selected', key='bottom-test-data-default'),
                                  html.Div([dcc.Dropdown(id='bottom-select-test-sample'),
                                            html.Img(id='bottom-test-sample-img', key='bottom-test-data-img')],
                                           hidden=True)]

        return dbc.Row(style={'marginTop': '10px', 'marginBottom': '20px'}, children=[
            # Input sample selection
            dbc.Col(md=2, align='start',
                    children=html.Div(className='detail-section', key='bottom-test-data',
                                      children=test_data_selector)),

            # Layer information
            dbc.Col(md=3, align='start',
                    children=html.Div(className='detail-section', key='bottom-layer',
                                      children=[html.Div(id='bottom-layer-title'),
                                                html.Div(id='bottom-layer-tabs', key='bottom-layer-tabs',
                                                         children=dummy_layer.get_layer_tabs())])
                    ),

            # Unit information
            dbc.Col(md=4, align='start',
                    children=html.Div(className='detail-section', key='bottom-unit',
                                      children=[html.Div(id='bottom-unit-title', key='bottom-unit-title',
                                                         children=html.H5('Maps', key='bottom-unit-title-text')),
                                                html.Div(id='bottom-unit-tabs', key='bottom-unit-tabs',
                                                         children=dummy_layer.get_unit_tabs(0))])),

            #  Maps
            dbc.Col(md=3, align='start',
                    children=html.Div(className='detail-section', key='bottom-maps',
                                      children=[html.Div(id='bottom-maps-title', key='bottom-maps-title',
                                                         children=html.H5('Maps', key='bottom-maps-title-text')),
                                                html.Div(id='bottom-maps-tabs', key='bottom-maps-tabs',
                                                         children=BottomPane._get_maps_tabs(None))]))
        ])

    def callbacks(self, app, model_sequence, grapher, test_data):
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

        @app.callback([Output('bottom-layer-title', 'children'), Output('bottom-layer-tabs', 'children'),
                       Output('bottom-unit-title', 'children'), Output('bottom-unit-tabs', 'children')],
                      [Input('center-selected-unit', 'data'), Input('top-epoch-index', 'data')],
                      [State('bottom-layer-tab-bar', 'active_tab'),
                       State('bottom-unit-tab-bar', 'active_tab')])
        def update_layer_info(selected_unit, _, active_layer_tab, active_unit_tab):
            """ Display selected layer/unit title and tabs """
            if selected_unit:
                layer = grapher.layers[selected_unit['layer_idx']]
            else:
                layer = AbstractLayer('dummy')
            layer_title = layer.get_layer_title()
            layer_tabs = layer.get_layer_tabs(active_layer_tab)
            unit_title = layer.get_unit_title(selected_unit['unit_idx'])
            unit_tabs = layer.get_unit_tabs(selected_unit['unit_idx'], active_unit_tab)
            return layer_title, layer_tabs, unit_title, unit_tabs

        @app.callback(Output('center-main-view', 'clickData'),
                      [Input('bottom-layer-figure', 'clickData'),
                       Input('bottom-maps-figure', 'clickData')],
                      [State('center-main-view', 'clickData')])
        def update_unit_selection(layer_click_data, activation_click_data, network_click_data):
            """ Click on the layer figure => update the main selection's unit """

            if network_click_data:
                click_data = None
                ctx = callback_context
                if ctx.triggered:
                    figure_id = ctx.triggered[0]['prop_id'].split('.')[0]

                    if figure_id == 'bottom-layer-figure':
                        click_data = layer_click_data
                    elif figure_id == 'bottom-maps-figure':
                        click_data = activation_click_data

                if click_data:
                    network_click_data['points'][0]['pointNumber'] = click_data['points'][0]['pointNumber']
            return network_click_data

        @app.callback([Output("bottom-layer-tab-content", "children"),
                       Output("bottom-layer-figure", "figure"),
                       Output("bottom-layer-tab-figure", "hidden")],
                      [Input("bottom-layer-tab-bar", "active_tab"),
                       Input('center-selected-unit', 'data')])
        def render_layer_tab_content(active_tab, selected_unit):
            """ layer info tab selected => update content """
            if active_tab and selected_unit is not None:
                layer = grapher.layers[selected_unit['layer_idx']]
                if layer is not None:
                    content, figure = layer.get_layer_tab_content(active_tab, selected_unit['unit_idx'])
                    return content, go.Figure() if figure is None else figure, figure is None
            return [], go.Figure(), True

        @app.callback([Output("bottom-unit-tab-content", "children"),
                       Output("bottom-unit-figure", "figure"),
                       Output("bottom-unit-tab-figure", "hidden")],
                      [Input("bottom-unit-tab-bar", "active_tab")],
                      [State('center-selected-unit', 'data')])
        def render_unit_tab_content(active_tab, selected_unit):
            """ layer info tab selected => update content """
            if active_tab and selected_unit is not None:
                layer = grapher.layers[selected_unit['layer_idx']]
                if layer is not None:
                    content, figure = layer.get_unit_tab_content(selected_unit['unit_idx'], active_tab)
                    return content, go.Figure() if figure is None else figure, figure is None
            return [], go.Figure(), True

        @app.callback([Output('bottom-maps-tab-content', 'children'),
                       Output('bottom-maps-figure', 'figure'),
                       Output('bottom-maps-tab-figure', 'hidden')],
                      [Input('bottom-maps-tab-bar', 'active_tab'),
                       Input('bottom-select-test-sample', 'value'),
                       Input('center-selected-unit', 'data'),
                       Input('top-epoch-index', 'data')])
        def update_activation_map(active_tab, index, selected_unit, _):
            if active_tab is not None \
                    and index is not None and test_data.x is not None \
                    and selected_unit and selected_unit['layer_idx'] is not None:
                if active_tab == 'activation':
                    if test_data.has_test_sample:
                        layer = grapher.layers[selected_unit['layer_idx']]
                        content, figure = layer.get_activation_map(model_sequence, test_data.x[index],
                                                                   selected_unit['unit_idx'])
                        return content, go.Figure() if figure is None else figure, figure is None
                    else:
                        return html.H5('Activation maps require test data'), go.Figure, False
            return [], go.Figure(), True

        return

    @staticmethod
    def _get_maps_tabs(active_tab: str = None):
        """ Tabs for the maps """
        return tabs.make('bottom-maps', {'activation': 'Activation'}, active_tab)
