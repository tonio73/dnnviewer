from dnnviewer.layers.Dense import Dense
from dnnviewer.Grapher import Grapher

import numpy as np


class TestGrapher:

    def test_clear_layers(self):

        grapher = Grapher()

        assert len(grapher.layers) == 0
        assert grapher.structure_props['num_dense'] == 0
        assert grapher.structure_props['num_convo2d'] == 0

        grads = weights = np.array([[1]])
        layer = Dense('test_1', 1, weights, grads)
        grapher.add_layer(layer)

        assert len(grapher.layers) == 1
        assert grapher.structure_props['num_dense'] == 1
        assert grapher.structure_props['num_convo2d'] == 0

        grapher.reset()

        assert len(grapher.layers) == 0
        assert grapher.structure_props['num_dense'] == 0
        assert grapher.structure_props['num_convo2d'] == 0
