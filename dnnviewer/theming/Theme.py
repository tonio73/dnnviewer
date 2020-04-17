from .SimpleColorScale import SimpleColorScale


class Theme:
    """ Manage the application theme """

    def __init__(self):
        self.plotly = 'plotly_dark'
        self.weight_color_scale = SimpleColorScale()
        self.gradient_color_scale = SimpleColorScale()
        self.bottom_figure_layout = dict(l=10, r=10, b=30, t=40)  # noqa: E741
