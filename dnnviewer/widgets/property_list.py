from typing import Dict
import dash_html_components as html


def widget(key, title, labels: Dict, values: Dict):
    """ Create a key value property list from two dictionaries with same keys """
    if len(values) > 0:
        return [html.H6(title, key=key),
                html.Ul([html.Li("%s: %s" % (labels[p], values[p]), key=key + p) for p in values.keys()])]
    return []
