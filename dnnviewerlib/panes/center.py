# Center pane of the application: network main view and properties

from dnnviewerlib.app import app, grapher, main_view

import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output

# Number of connections to show at init
topn_init = 3


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
