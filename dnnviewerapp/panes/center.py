#
# Center pane of the application: network main view and properties
#

from dnnviewerapp import app, grapher, test_data

import plotly.graph_objects as go
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output

# Number of connections to show at init
topn_init = 3

# Main network view
main_view = go.Figure()


def render():
    """ Prepare graphical structures before Dash rendering """

    main_view.update_layout(margin=dict(l=10, r=10, b=30, t=30))

    grapher.plot_layers(main_view)

    # Init initially shown connections based on the first test sample
    if test_data.has_test_sample:
        grapher.plot_topn_connections(main_view, 3, grapher.layers[-1],
                                      test_data.y_test[test_data.test_sample_init])


def get_layout():
    """ Get pane layout """
    return dbc.Row([
        dbc.Col(dcc.Graph(id='network-view', figure=main_view, config=dict(scrollZoom=True), animate=True), md=9),
        dbc.Col([html.Label('Show top n connections:'),
                 dcc.Slider(id='topn-criteria',
                            min=1.0, max=4.0, step=1.0, value=topn_init,
                            marks={str(d): str(d) for d in range(0, 5)})
                 ], md=3)
    ])


def callbacks():
    """ Local callbacks """

    @app.callback(Output('network-view', 'figure'),
                  [Input('topn-criteria', 'value'), Input('network-view', 'clickData')])
    def update_figure(topn, click_data):
        """ Update the main view when some unit is selected or the number of connections to show is changed """
        if click_data:
            layer, unit_idx = grapher.get_layer_unit_from_click_data(click_data)
            grapher.plot_topn_connections(main_view, topn, layer, unit_idx)
        return main_view

    return
