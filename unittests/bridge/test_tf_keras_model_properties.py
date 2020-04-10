from dnnviewer.bridge import tensorflow as btf
from dnnviewer.Grapher import Grapher

from tensorflow import keras


def test_keras_loss_string():
    """ Keras loss set as string like 'categorical_crossentropy' """

    grapher = Grapher()

    model = keras.models.Sequential()
    model.compile(loss='categorical_crossentropy')

    btf.keras_set_model_properties(grapher, model)

    assert grapher.training_props['loss'] == 'Categorical cross-entropy'


def test_keras_loss_class():
    """ Keras loss using calss instance """

    grapher = Grapher()

    model = keras.models.Sequential()
    model.compile(loss=keras.losses.MeanSquaredError())

    btf.keras_set_model_properties(grapher, model)

    assert grapher.training_props['loss'] == 'Mean squared error'


def test_keras_loss_custom_class():
    """ Keras loss built out of a custom class """

    class CustomLoss:

        def __init__(self):
            pass

    grapher = Grapher()

    model = keras.models.Sequential()
    model.compile(loss=CustomLoss())

    btf.keras_set_model_properties(grapher, model)

    assert grapher.training_props['loss'] == 'CustomLoss'


def test_keras_loss_func():
    """ Keras loss set as standard function """

    grapher = Grapher()

    model = keras.models.Sequential()
    model.compile(loss=keras.losses.hinge)

    btf.keras_set_model_properties(grapher, model)

    assert grapher.training_props['loss'] == 'Hinge'


def test_keras_loss_custom_func():
    def custom_loss(y_true, y_est):
        return y_true == y_est

    grapher = Grapher()

    model = keras.models.Sequential()
    model.compile(loss=custom_loss)

    btf.keras_set_model_properties(grapher, model)

    assert grapher.training_props['loss'] == 'custom_loss'
