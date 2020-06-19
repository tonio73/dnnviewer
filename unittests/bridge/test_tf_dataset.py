"""
    Tests in this file are more of integration tests since they access for download TensorFlow Dataset and related data
"""

from dnnviewer.TestData import TestData
from dnnviewer.bridge import ModelError
from dnnviewer.bridge.tensorflow_datasets import load_test_data
from dnnviewer.bridge.tensorflow import keras_prepare_input, keras_prepare_labels

from tensorflow import keras
import numpy as np
import pytest


def test_tf_dataset_load_mnist():
    """ Load MNIST dataset (quick)"""
    test_data = TestData()
    load_test_data('mnist', test_data, 128)

    assert test_data.has_test_sample is True
    assert np.array_equal(test_data.x.shape, [128, 28, 28, 1])
    assert np.array_equal(test_data.y.shape, [128])


def test_tf_dataset_load_prepare_fashion_mnist():
    """ Load Fashion MNIST dataset (quick), prepare for model with 32x32 input """
    test_data = TestData()
    load_test_data('fashion_mnist', test_data, 128)

    assert test_data.has_test_sample is True
    assert np.array_equal(test_data.x.shape, [128, 28, 28, 1])
    assert test_data.x.dtype is np.dtype('|u1'), "uint8"
    assert np.array_equal(test_data.y.shape, [128])
    assert test_data.y.dtype is np.dtype('<i8'), "int64"

    test_data.x_format = keras_prepare_input(np.float, np.array([None, 32, 32, 1]), test_data.x)

    assert np.array_equal(test_data.x_format.shape, [128, 32, 32, 1])
    assert test_data.x_format.dtype is np.dtype('<f8'), "float32"


def test_tf_dataset_load_prepare_fashion_mnist_incompatible_shape():
    """ Load Fashion MNIST dataset (quick), attempt to prepare for input with less dimensions """
    test_data = TestData()
    load_test_data('fashion_mnist', test_data, 128)

    with pytest.raises(ModelError):
        test_data.x_format = keras_prepare_input(np.float, np.array([None, 32]), test_data.x)


def test_tf_keras_prepare_labels():
    """ Prepare labels in case of categorical crossentropy loss """

    test_data = TestData()
    load_test_data('fashion_mnist', test_data, 128)

    assert np.array_equal(test_data.y.shape, [128])
    assert test_data.y.dtype is np.dtype('<i8'), "int32"

    model = keras.models.Sequential([
        keras.layers.Dense(10)
    ])
    model.compile(loss=keras.losses.CategoricalCrossentropy(from_logits=False))

    prepared_labels = keras_prepare_labels(model, test_data.y)

    assert np.array_equal(prepared_labels.shape, [128, 10])
    assert prepared_labels.dtype is np.dtype('<f4'), "float32"


def test_tf_keras_prepare_labels_unneeded():
    """ Prepare labels in case unneeded """

    test_data = TestData()
    load_test_data('fashion_mnist', test_data, 128)

    assert np.array_equal(test_data.y.shape, [128])
    assert test_data.y.dtype is np.dtype('<i8'), "int32"

    model = keras.models.Sequential([
        keras.layers.Dense(10)
    ])
    model.compile(loss=keras.losses.BinaryCrossentropy())

    prepared_labels = keras_prepare_labels(model, test_data.y)

    assert np.array_equal(prepared_labels.shape, [128]), "unchanged from input labels"
