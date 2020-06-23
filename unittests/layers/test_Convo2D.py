import numpy as np

from dnnviewer.layers.Dense import Dense
from dnnviewer.layers.Convo2D import Convo2D


class TestConvo2D:

    def test_plot_topn_connections_backward(self):
        """ Backward top 2 connections """

        # 3 x 3 shape filters x 2 with 3 inputs => weights array of shape (3, 3, 2, 3)
        weights = self.weights_convo_3_3_4_3
        layer = Convo2D('test_1', '', 3, weights, weights)
        layer.set_coordinates(10, 0)

        prev_layer = Dense('test_prev', '', 4, np.ones((6, 4)), np.ones((6, 4)))  # weights/grads of other do no matter
        prev_layer.set_coordinates(0, 0)

        strongest_idx, shapes = layer.plot_topn_connections_backward(prev_layer, 2, [1, 2])

        assert strongest_idx.shape == (2,)
        assert (strongest_idx == np.array([1, 3])).all()
        assert len(shapes) == 4  # Each Convolution to the top Dense
        assert isinstance(shapes[0], dict)
        assert shapes[0]['type'] == 'path'

    def test_plot_topn_connections_backward_flatten(self):
        """ Backward top 2 connections with flatten """

        # 3 x 3 shape filters x 4 with 3 inputs => weights array of shape (3, 3, 2, 3)
        weights = self.weights_convo_3_3_4_3
        layer = Convo2D('test_1', '', 3, weights, weights)
        layer.set_coordinates(10, 0)
        layer.flatten_output = True  # <--

        prev_layer = Dense('test_prev', '', 4, np.ones((6, 4)), np.ones((6, 4)))  # weights/grads of other do no matter
        prev_layer.set_coordinates(0, 0)
        prev_layer.set_coordinates(0, 0)

        strongest_idx, shapes = layer.plot_topn_connections_backward(prev_layer, 2, [1, 2])

        assert strongest_idx.shape == (2,)
        assert (strongest_idx == np.array([1, 3])).all()
        assert len(shapes) == 4  # Each Convolution to the top Dense
        assert isinstance(shapes[0], dict)
        assert shapes[0]['type'] == 'path'

    def test_plot_topn_connections_backward_flatten_stride(self):
        """ Backward top 2 connections with stride and flatten """

        # 3 x 3 shape filters x 4 with 3 inputs => weights array of shape (3, 3, 4, 3)
        weights = self.weights_convo_3_3_4_3
        layer = Convo2D('test_1', '', 3, weights, weights, flatten_output=True)  # <--
        layer.set_coordinates(10, 0)

        # Stride or pooling induce a fractional sampling factor (down-sampling)
        layer.append_sampling_factor(1 / np.array([2, 1, 1, 1]))  # <--

        prev_layer = Dense('test_prev', '', 4, np.ones((6, 4)), np.ones((6, 4)))  # weights/grads of other do no matter
        prev_layer.set_coordinates(0, 0)

        strongest_idx, shapes = layer.plot_topn_connections_backward(prev_layer, 2, [1, 2])

        assert strongest_idx.shape == (2,)
        assert (strongest_idx == np.array([1, 3])).all()   # <--
        assert len(shapes) == 4  # Each Convolution to the top Dense
        assert isinstance(shapes[0], dict)
        assert shapes[0]['type'] == 'path'

    def test_plot_topn_connections_backward_flatten_stride2d(self):
        """ Backward top 2 connections with stride on 2D and flatten """

        # 3 x 3 shape filters x 4 with 3 inputs => weights array of shape (3, 3, 4, 3)
        weights = self.weights_convo_3_3_4_3
        layer = Convo2D('test_1', '', 3, weights, weights, flatten_output=True)  # <--
        layer.set_coordinates(10, 0)

        # Stride or pooling induce a fractional sampling factor (down-sampling)
        layer.append_sampling_factor(1 / np.array([2, 2, 1, 1]))  # <--

        prev_layer = Dense('test_prev', '', 4, np.ones((6, 4)), np.ones((6, 4)))  # weights/grads of other do no matter
        prev_layer.set_coordinates(0, 0)

        strongest_idx, shapes = layer.plot_topn_connections_backward(prev_layer, 2, [1, 2])

        assert strongest_idx.shape == (2,)
        assert (strongest_idx == np.array([1, 3])).all()   # <--
        assert len(shapes) == 4  # Each Convolution to the top Dense
        assert isinstance(shapes[0], dict)
        assert shapes[0]['type'] == 'path'

    def test_plot_topn_connections_forward(self):
        """ Forward top 2 connections """

        weights = np.array([[[[-1, 0, 1], [+0, 0, 1]], [[1, 1, 2], [2, 3, 3]], [[7, 1, 2], [2, 3, 4]]],
                            [[[+1, 1, 2], [-1, 1, 2]], [[1, 0, 6], [4, 1, 8]], [[6, 1, 2], [2, 0, 2]]],
                            [[[-3, 0, 5], [+0, 1, 2]], [[0, 2, 4], [5, 1, 4]], [[9, 1, 2], [4, 6, 6]]]])

        layer = Convo2D('test_2', '', 3, weights, weights)
        layer.set_coordinates(10, 0)
        prev_layer = Dense('test_prev', '', 2, np.ones((3, 2)), np.ones((3, 2)))  # weights/grads of other do no matter
        prev_layer.set_coordinates(0, 0)

        strongest_idx, shapes = layer.plot_topn_connections_forward(prev_layer, 2, [1])

        assert strongest_idx.shape == (2,)
        assert (strongest_idx == np.array([1, 2])).all()
        assert len(shapes) == 2  # For each active Dense to each Convolution top 2
        assert isinstance(shapes[0], dict)
        assert shapes[0]['type'] == 'path'

    weights_convo_3_3_2_3 = np.array([[[[-1, 1, 0], [+0, 1, 0]], [[1, 2, 1], [2, 3, 3]], [[7, 2, 1], [2, 4, 3]]],
                                      [[[+1, 2, 1], [-1, 2, 1]], [[1, 6, 0], [4, 8, 1]], [[6, 2, 1], [2, 2, 0]]],
                                      [[[-3, 5, 0], [+0, 2, 1]], [[0, 4, 2], [5, 4, 1]], [[9, 2, 1], [4, 6, 6]]]])

    # Max on convolution filters
    # [[9, 2, 6], [5, 6, 8], [10, 2, 6], [5, 6, 11]]
    weights_convo_3_3_4_3 = np.array([[[[-1, 1, 0], [+0, 1, 0], [-1, 1, 0], [+0, 1, 0]],
                                       [[1, 2, 1], [2, 3, 3], [1, 2, 1], [2, 3, 3]],
                                       [[7, 2, 1], [2, 4, 3], [7, 2, 1], [2, 4, 3]]],
                                      [[[+1, 2, 1], [-1, 2, 1], [+1, 2, 1], [-1, 2, 1]],
                                       [[1, 6, 0], [4, 8, 1], [1, 6, 0], [4, 11, 1]],
                                       [[6, 2, 1], [2, 2, 0], [6, 2, 1], [2, 2, 0]]],
                                      [[[-3, 5, 0], [+0, 2, 1], [-3, 5, 0], [+0, 2, 1]],
                                       [[0, 4, 2], [5, 4, 1], [0, 4, 2], [5, 4, 1]],
                                       [[9, 2, 1], [4, 6, 6], [10, 2, 1], [4, 6, 6]]]])