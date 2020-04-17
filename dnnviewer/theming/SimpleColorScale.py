import numpy as np
from plotly.colors import diverging


class SimpleColorScale:
    """ Simple color selection from a sequence """

    def __init__(self, scale_seq=diverging.RdBu_r, mini=-1, maxi=1):
        self.seq = scale_seq
        self.min = mini
        self.max = maxi
        self.step = (maxi - mini) / len(scale_seq)

    def get_color(self, values):
        """Get the color(s) corresponding to value(s) """
        if isinstance(values, (int, float)):
            return self.seq[np.clip(int((values - self.min) / self.step), 0, len(self.seq) - 1)]
        else:
            idx = np.clip(np.floor((values - self.min) / self.step).astype(np.int), 0, len(self.seq) - 1)
            return [self.seq[c] for c in idx]

    def as_dict(self, value=None):
        """ Export as a dictionary to be used as colorscale argument in plotly """
        if value is None:
            return dict(colorscale=self.seq, cmin=self.min, cmax=self.max)
        else:
            return dict(color=value, colorscale=self.seq, cmin=self.min, cmax=self.max)

    def min_color(self):
        return self.seq[0]

    def max_color(self):
        return self.seq[-1]
