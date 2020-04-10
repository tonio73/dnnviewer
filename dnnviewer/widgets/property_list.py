from typing import Dict
import dash_html_components as html


def widget(title, labels: Dict, values: Dict):
    """ Create a key value property list from two dictionaries with same keys """
    return [html.H6(title),
            html.Ul([html.Li("%s: %s" % (labels[p], values[p])) for p in labels.keys()])]
