

class Connector:
    """ Single connector between to units of different layer OR collection of such connectors """

    def __init__(self, layer_from, layer_to, unit_from, unit_to, amplitude, color_scale):
        self.layer_from = layer_from
        self.layer_to = layer_to
        self.unit_from = unit_from
        self.unit_to = unit_to
        self.amplitude = amplitude
        self.color_scale = color_scale
        #
        self.control_delta = 0.2

    def get_shapes(self):
        x_from, y_from = self.layer_from.get_unit_position(self.unit_from, True)
        x_to, y_to = self.layer_to.get_unit_position(self.unit_to, False)

        # Bezier support through SVG in plotly : https://plot.ly/python/shapes/#svg-paths
        # SVG Bezier path : https://developer.mozilla.org/en-US/docs/Web/SVG/Tutorial/Paths
        colors = self.color_scale.get_color(self.amplitude)
        paths = ['M %f, %f C %f, %f %f, %f %f, %f' % (x0, y0,
                                                      x0 + self.control_delta, y0,
                                                      x1 - self.control_delta, y1,
                                                      x1, y1) for x0, y0, x1, y1 in zip(x_from, y_from, x_to, y_to)]
        return [dict(type='path', path=p, line=dict(color=c, width=1.5), opacity=1.0, layer='below')
                for p, c in zip(paths, colors)]
