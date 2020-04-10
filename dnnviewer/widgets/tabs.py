import dash_html_components as html
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
            html.Div(id=prefix + "-tab-content", className="p-2 detail-tab border-left",
                     children=default_tab_content)]
