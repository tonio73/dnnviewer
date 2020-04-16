#
# Center pane of the application: network main view and properties
#

from . import AbstractPane

import plotly.graph_objects as go
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
from dash.exceptions import PreventUpdate

import logging


class CenterPane(AbstractPane):
    # Main network view
    main_view = None

    def render(self, grapher):
        """ Prepare graphical structures before Dash rendering """
        self.main_view = go.Figure()
        self.main_view.update_layout(margin=dict(l=10, r=10, b=30, t=30), autosize=True)  # noqa: E741
        grapher.plot_layers(self.main_view)

    def get_layout(self, model_sequence, grapher, test_data):
        """ Get pane layout """

        # Initialize the selected unit to the output corresponding to the selected test data
        if test_data.has_test_sample and len(grapher.layers) > 0:
            selected_unit = dict(layer_idx=len(grapher.layers)-1, unit_idx=test_data.y[0])
        else:
            selected_unit = None

        graph_config = dict(scrollZoom=True, displaylogo=False,
                            modeBarButtonsToRemove=['lasso2d', 'resetScale2d', 'toggleSpikelines',
                                                    'hoverClosestCartesian', 'hoverCompareCartesian'])

        return dbc.Row([
            dcc.Store(id='center-selected-unit', data=selected_unit),
            dbc.Col(dcc.Graph(id='center-main-view', figure=self.main_view, config=graph_config,
                              animate=True, responsive=True), md=9),
            dbc.Col(html.Div(className='detail-section',
                             children=[
                                 dcc.Store(id='center-topn-criteria', data=grapher.topn_init),
                                 html.Div(id='central-model-title',
                                          children=html.H5(grapher.name if grapher.name else "Model")),
                                 html.Div(id='center-model-tabs',
                                          children=grapher.get_model_tabs(None))
                             ]), md=3)
        ])

    def callbacks(self, app, model_sequence, grapher, test_data):
        """ Dash callbacks """

        logger = logging.getLogger(__name__)

        @app.callback(Output('center-selected-unit', 'data'),
                      [Input('center-main-view', 'clickData')])
        def select_unit(click_data):
            if click_data:
                point = click_data['points'][0]
                return dict(layer_idx=int(point['curveNumber']),
                            unit_idx=point['pointNumber'])
            raise PreventUpdate

        @app.callback(Output("center-model-tab-content", "children"),
                      [Input("center-model-tab-bar", "active_tab"),
                       Input('top-epoch-index', 'data')])
        def render_unit_tab_content(active_tab, _):
            """ model info tab selected => update content """
            return grapher.get_model_tab_content(active_tab)

        @app.callback(Output('center-main-view', 'figure'),
                      [Input('center-topn-criteria', 'data'),
                       Input('center-selected-unit', 'data'),
                       Input('top-epoch-index', 'data')])
        def update_figure(topn: int, selected_unit, epoch_index):
            """ Update the main view when some unit is selected or the number of connections to show is changed """
            if topn is None:  # The slide might be hidden => 'center-topn-criteria' store is not initialized
                topn = grapher.topn_init

            if epoch_index is None:  # In case of single model sequence, the epoch index might be uninitialized
                epoch_index = 0

            if selected_unit is None:
                logger.warning('update_figure prevent update since selected unit is %s', selected_unit)
                raise PreventUpdate

            if selected_unit:
                logger.debug(f"update_figure: selected_unit{selected_unit}")
                grapher.plot_topn_connections(self.main_view, int(topn),
                                              selected_unit['layer_idx'], selected_unit['unit_idx'])
            return self.main_view

        @app.callback(Output('center-topn-criteria', 'data'),
                      [Input('center-topn-criteria-slider', 'value')])
        def update_topn_criteria(value):
            return value

        return
