import numpy as np

from dnnviewerlib.layers.Dense import Dense


class TestDense:

    def test_plot_topn_connections_backward(self):
        # def plot_topn_connections(self, backward_layer, topn, active_units, True)

        weights = np.array([[1, 2, 3, 4], [2, 6, 8, 2], [5, 4, 4, 6]])
        layer = Dense('test_1', 4, weights, plotly_theme='plotly_dark')
        layer.set_xoffset(10)
        prev_layer = Dense('test_prev', 3, np.ones((2, 3)), plotly_theme='plotly_dark')  # weight of other do no matter
        prev_layer.set_xoffset(0)

        strongest_idx, shapes = layer.plot_topn_connections(prev_layer, 2, [2, 3], True)

        assert strongest_idx.shape == (3,)
        assert (strongest_idx == np.array([0, 1, 2])).all()
        assert len(shapes) == 4
        assert isinstance(shapes[0], dict)
        assert shapes[0]['type'] == 'path'

    def test_plot_topn_connections_forward(self):
        # def plot_topn_connections(self, backward_layer, topn, active_units, False)

        weights = np.array([[1, 2, 3, 4], [2, 6, 8, 2], [5, 4, 4, 6]])
        layer = Dense('test_1', 4, weights, plotly_theme='plotly_dark')
        layer.set_xoffset(10)
        prev_layer = Dense('test_prev', 3, np.ones((2, 3)), plotly_theme='plotly_dark')  # weight of other do no matter
        prev_layer.set_xoffset(0)

        strongest_idx, shapes = layer.plot_topn_connections(prev_layer, 2, [1, 2], False)

        assert strongest_idx.shape == (4,)
        assert (strongest_idx == np.array([0, 1, 2, 3])).all()
        assert len(shapes) == 4
        assert isinstance(shapes[0], dict)
        assert shapes[0]['type'] == 'path'
