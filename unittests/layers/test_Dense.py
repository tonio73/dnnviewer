from dnnviewer.layers.Dense import Dense

import numpy as np


class TestDense:

    def test_plot_topn_connections_backward(self):

        grads = weights = np.array([[1, 2, 3, 4], [2, 6, 8, 2], [5, 4, 4, 6]])
        layer = Dense('test_1', '', 4, weights, grads)
        layer.set_coordinates(10, 0)
        prev_layer = Dense('test_prev', '', 3, np.ones((2, 3)), np.ones((2, 3)))  # weight/grads of other do no matter
        prev_layer.set_coordinates(0, 0)

        strongest_idx, shapes = layer.plot_topn_connections_backward(prev_layer, 2, [2, 3])

        assert strongest_idx.shape == (3,)
        assert (strongest_idx == np.array([0, 1, 2])).all()
        assert len(shapes) == 4
        assert isinstance(shapes[0], dict)
        assert shapes[0]['type'] == 'path'

    def test_plot_topn_connections_forward(self):

        grads = weights = np.array([[1, 2, 3, 4], [2, 6, 8, 2], [5, 4, 4, 6]])
        layer = Dense('test_1', '', 4, weights, grads)
        layer.set_coordinates(10, 0)
        prev_layer = Dense('test_prev', '', 3, np.ones((2, 3)), np.ones((2, 3)))  # weight/grads of other do no matter
        prev_layer.set_coordinates(0, 0)

        strongest_idx, shapes = layer.plot_topn_connections_forward(prev_layer, 2, [1, 2])

        assert strongest_idx.shape == (4,)
        assert (strongest_idx == np.array([0, 1, 2, 3])).all()
        assert len(shapes) == 4
        assert isinstance(shapes[0], dict)
        assert shapes[0]['type'] == 'path'
