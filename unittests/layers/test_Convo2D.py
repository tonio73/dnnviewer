import numpy as np

from dnnviewer.layers.Dense import Dense
from dnnviewer.layers.Convo2D import Convo2D


class TestConvo2D:

    def test_plot_topn_connections_backward(self):
        """ Backward top 2 connections """
        # def plot_topn_connections(self, backward_layer, topn, active_units, True)

        weights = np.array([[[[-1, 0, 1], [+0, 0, 1]], [[1, 1, 2], [2, 3, 3]], [[7, 1, 2], [2, 3, 4]]],
                            [[[+1, 1, 2], [-1, 1, 2]], [[1, 0, 6], [4, 1, 8]], [[6, 1, 2], [2, 0, 2]]],
                            [[[-3, 0, 5], [+0, 1, 2]], [[0, 2, 4], [5, 1, 4]], [[9, 1, 2], [4, 6, 6]]]])
        # Max on convolution filters
        # [[9, 2, 6], [5, 6, 8]]
        layer = Convo2D('test_1', 3, weights, plotly_theme='plotly_dark')
        layer.set_xoffset(10)

        prev_layer = Dense('test_prev', 2, np.ones((3, 2)), 'plotly_dark')  # weights of other do no matter
        prev_layer.set_xoffset(0)

        strongest_idx, shapes = layer.plot_topn_connections(prev_layer, 2, [1, 2], True)

        assert strongest_idx.shape == (2,)
        assert (strongest_idx == np.array([0, 1])).all()
        assert len(shapes) == 4  # Each Convolution to the top Dense
        assert isinstance(shapes[0], dict)
        assert shapes[0]['type'] == 'path'

    def test_plot_topn_connections_forward(self):
        """ Forward top 2 connections """
        # def plot_topn_connections(self, backward_layer, topn, active_units, False)

        weights = np.array([[[[-1, 0, 1], [+0, 0, 1]], [[1, 1, 2], [2, 3, 3]], [[7, 1, 2], [2, 3, 4]]],
                            [[[+1, 1, 2], [-1, 1, 2]], [[1, 0, 6], [4, 1, 8]], [[6, 1, 2], [2, 0, 2]]],
                            [[[-3, 0, 5], [+0, 1, 2]], [[0, 2, 4], [5, 1, 4]], [[9, 1, 2], [4, 6, 6]]]])

        layer = Convo2D('test_2', 3, weights, plotly_theme='plotly_dark')
        layer.set_xoffset(10)
        prev_layer = Dense('test_prev', 2, np.ones((3, 2)), plotly_theme='plotly_dark')  # weights of other do no matter
        prev_layer.set_xoffset(0)

        strongest_idx, shapes = layer.plot_topn_connections(prev_layer, 2, [1], False)

        assert strongest_idx.shape == (2,)
        assert (strongest_idx == np.array([0, 2])).all()
        assert len(shapes) == 2  # For each active Dense to each Convolution top 2
        assert isinstance(shapes[0], dict)
        assert shapes[0]['type'] == 'path'
