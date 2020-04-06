#
# Center pane of the application: network main view and properties
#

from .. import app, grapher, test_data

import plotly.graph_objects as go
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output

# Main network view
main_view = go.Figure()


def render():
    """ Prepare graphical structures before Dash rendering """

    main_view.update_layout(margin=dict(l=10, r=10, b=30, t=30))

    grapher.plot_layers(main_view)

    # Init initially shown connections based on the first test sample
    if test_data.has_test_sample:
        grapher.plot_topn_connections(main_view, 3, grapher.layers[-1], 0)


def get_layout():
    """ Get pane layout """
    return dbc.Row([
        dbc.Col(dcc.Graph(id='center-main-view', figure=main_view, config=dict(scrollZoom=True), animate=True), md=9),
        dbc.Col(html.Div(className='detail-section',
                         children=[
                             dcc.Store(id='center-topn-criteria', data=grapher.topn_init),
                             html.Div(id='central-model-title',
                                      children=html.H5(grapher.name if grapher.name else "Model")),
                             html.Div(id='center-model-tabs',
                                      children=grapher.get_model_tabs(None))
                         ]), md=3)
    ])


def callbacks():
    """ Local callbacks """

    @app.callback(
        Output("center-model-tab-content", "children"),
        [Input("center-model-tab-bar", "active_tab")])
    def render_unit_tab_content(active_tab):
        """ model info tab selected => update content """
        return grapher.get_model_tab_content(active_tab)

    @app.callback(Output('center-main-view', 'figure'),
                  [Input('center-topn-criteria', 'data'), Input('center-main-view', 'clickData')])
    def update_figure(topn: int, click_data):
        """ Update the main view when some unit is selected or the number of connections to show is changed """
        if topn is None:  # The slide might be hidden => store is not initialized
            topn = grapher.topn_init

        if click_data:
            layer, unit_idx = grapher.get_layer_unit_from_click_data(click_data)
            grapher.plot_topn_connections(main_view, int(topn), layer, unit_idx)
        return main_view

    @app.callback(Output('center-topn-criteria', 'data'),
                  [Input('center-topn-criteria-slider', 'value')])
    def update_topn_criteria(value):
        return value

    return
