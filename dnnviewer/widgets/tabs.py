import dash_html_components as html
import dash_core_components as dcc
import dash_bootstrap_components as dbc


def make(prefix: str, tab_def, previous_active: str = None, default_tab_content=None):
    """ Create tab bar and container for the layer information sub-panel """
    if previous_active is not None and previous_active in list(tab_def.keys()):
        active_tab = previous_active
    elif len(tab_def) > 0:
        active_tab = list(tab_def)[0]
    else:
        active_tab = ''

    return [dbc.Tabs(id=prefix + "-tab-bar", active_tab=active_tab,
                     children=[dbc.Tab(label=tab_def[t], tab_id=t) for t in tab_def]),
            html.Div(className="p-2 detail-tab border-left", key=prefix + "-tab-wrap",
                     children=[
                         html.Div(id=prefix + "-tab-content", key=prefix + "-tab-content-wrap",
                                  children=default_tab_content),
                         html.Div(id=prefix + "-tab-figure", hidden=True, key=prefix + "-tab-figure-wrap",
                                  children=dcc.Graph(id=prefix + '-figure',
                                                     config=_get_graph_config()))
                     ])]


def _get_graph_config():
    """" Graph config for all detail bottom figures """
    return dict(scrollZoom=True, displaylogo=False,
                modeBarButtonsToRemove=['lasso2d', 'resetScale2d', 'zoomIn2d', 'zoomOut2d',
                                        'toggleSpikelines', 'select2d',
                                        'hoverClosestCartesian', 'hoverCompareCartesian'])
