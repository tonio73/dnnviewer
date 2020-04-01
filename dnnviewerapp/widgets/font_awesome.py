import dash_html_components as html


CDN_CSS_URL = 'https://use.fontawesome.com/releases/v5.7.0/css/all.css'


def icon(name):
    return html.I(className="fas fa-" + name)
