from .SimpleColorScale import SimpleColorScale

from plotly.colors import diverging, sequential

# Float format to apply when printing
float_fmt = '%.4g'


class Theme:
    """ Manage the application theme """

    def __init__(self):
        self.plotly = 'plotly_dark'
        self.weight_color_scale = SimpleColorScale(diverging.RdBu_r, -1, 1)
        self.gradient_color_scale = SimpleColorScale(diverging.BrBG, -2, 2)
        self.activation_color_scale = SimpleColorScale(sequential.Cividis)
        self.bottom_figure_margins = dict(l=10, r=10, b=30, t=40)  # noqa: E741
