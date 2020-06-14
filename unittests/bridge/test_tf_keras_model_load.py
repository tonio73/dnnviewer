from dnnviewer.bridge.KerasNetworkExtractor import KerasNetworkExtractor
from dnnviewer.bridge import ModelError
from dnnviewer.Grapher import Grapher
from dnnviewer.layers.Dense import Dense
from dnnviewer.layers.Convo2D import Convo2D
from dnnviewer.layers.Input import Input
from dnnviewer.TestData import TestData

from tensorflow import keras

import numpy as np
import pytest


def test_model_not_sequential():
    """ Only Keras Sequential model is supported """

    grapher = Grapher()
    model = keras.models.Model()

    with pytest.raises(ModelError):
        extractor = KerasNetworkExtractor(grapher, model, TestData())
        extractor.process()


def test_model_empty():
    """ Load an empty sequential model => should return grapher with no layer """

    grapher = Grapher()
    model = keras.models.Sequential()

    extractor = KerasNetworkExtractor(grapher, model, TestData())
    extractor.process()
    # Should raise en error in the logger

    assert len(grapher.layers) == 0


def test_model_1_dense():
    """ Model with 1 Dense layer """

    grapher = Grapher()
    model = keras.models.Sequential(keras.layers.Dense(32, input_shape=[10], name='my_dense'))
    model.compile(loss='categorical_crossentropy')
    model.set_weights([np.ones((10, 32)), np.zeros(32)])

    extractor = KerasNetworkExtractor(grapher, model, TestData())
    extractor.process()

    # 1 layer added for input
    assert len(grapher.layers) == 2

    layer_input = grapher.layers[0]
    assert isinstance(layer_input, Input)
    assert layer_input.num_unit == 10

    layer_dense = grapher.layers[1]
    assert isinstance(layer_dense, Dense)
    assert layer_dense.name == 'my_dense'
    assert layer_dense.num_unit == 32


def test_model_1_convo2d():
    """ Model with 1 Conv2D layer """

    grapher = Grapher()
    model = keras.models.Sequential(keras.layers.Conv2D(128, kernel_size=(3, 3), input_shape=[32, 32, 3],
                                                        name='my_conv'))
    model.compile(loss='categorical_crossentropy')
    model.set_weights([np.ones((3, 3, 3, 128)), np.zeros(128)])

    extractor = KerasNetworkExtractor(grapher, model, TestData())
    extractor.process()

    # 1 layer added for input
    assert len(grapher.layers) == 2

    layer_input = grapher.layers[0]
    assert isinstance(layer_input, Input)
    assert layer_input.num_unit == 3

    layer_convo = grapher.layers[1]
    assert isinstance(layer_convo, Convo2D)
    assert layer_convo.name == 'my_conv'
    assert layer_convo.num_unit == 128
