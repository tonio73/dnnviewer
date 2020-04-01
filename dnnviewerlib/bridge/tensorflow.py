""" Method to create graph view of a TensorFlow model """

from tensorflow import keras

from ..layers import Dense, Convo2D, Input, AbstractLayer
from ..SimpleColorScale import SimpleColorScale
from .AbstractActivationMapper import AbstractActivationMapper

import numpy as np


def keras_load_sequential_network(grapher, model_file_name, input_classes=None, output_classes=None):
    """ Load and extract graphical representation of a Keras sequential model """

    model0 = keras.models.load_model(model_file_name)
    # Create all other layers from the Keras Sequential model
    keras_extract_sequential_network(grapher, model0, input_classes, output_classes)
    return KerasActivationMapper(model0)


def keras_extract_sequential_network(grapher, model, input_classes=None, output_classes=None):
    """ Create a graphical representation of a Keras Sequential model's layers """

    if not isinstance(model, keras.models.Sequential):
        print("Unexpected model of type '%s', expecting Sequential model" % type(model))

    if len(model.layers) == 0:
        print("Empty model")
        return

    plotly_theme = grapher.plotly_theme
    color_scale = SimpleColorScale()

    previous_layer = None

    # Input placeholder if first layer is a layer with weights
    if len(model.layers[0].get_weights()) > 0:
        input_dim = model.layers[0].get_weights()[0].shape[-2]
        if input_classes is not None and len(input_classes) != input_dim:
            print("Wrong length of input classes, got %d, expecting %d" % (len(input_classes), input_dim))
            input_classes = None

        input_layer = Input('input', input_dim, plotly_theme, input_classes)
        grapher.add_layer(input_layer)
        previous_layer = input_layer

    for keras_layer in model.layers:
        if isinstance(keras_layer, keras.layers.Dense):
            layer = Dense(keras_layer.name, keras_layer.output_shape[-1], keras_layer.weights[0].numpy(),
                          plotly_theme, color_scale)
            grapher.add_layer(layer)
            previous_layer = layer

        elif isinstance(keras_layer, keras.layers.Conv2D):
            layer = Convo2D(keras_layer.name, keras_layer.output_shape[-1], keras_layer.weights[0].numpy(),
                            plotly_theme, color_scale)
            grapher.add_layer(layer)
            previous_layer = layer

        elif isinstance(keras_layer, keras.layers.Flatten):
            if isinstance(previous_layer, Convo2D):
                previous_layer.flatten_output = True
            elif previous_layer is None:
                # Input layer
                input_dim = keras_layer.get_output_shape_at(0)[-1]
                input_layer = Input('input', input_dim, plotly_theme, input_classes)
                grapher.add_layer(input_layer)
                previous_layer = input_layer

        elif isinstance(keras_layer, (keras.layers.ActivityRegularization, keras.layers.Dropout,
                                      keras.layers.SpatialDropout1D, keras.layers.SpatialDropout2D,
                                      keras.layers.SpatialDropout3D,
                                      keras.layers.Activation,
                                      keras.layers.MaxPooling1D, keras.layers.MaxPooling2D, keras.layers.MaxPooling3D,
                                      keras.layers.AveragePooling1D, keras.layers.AveragePooling2D,
                                      keras.layers.AveragePooling3D,
                                      keras.layers.GlobalAveragePooling1D, keras.layers.GlobalAveragePooling2D,
                                      keras.layers.GlobalAveragePooling3D,
                                      keras.layers.GlobalMaxPooling1D, keras.layers.GlobalMaxPooling2D,
                                      keras.layers.GlobalMaxPooling3D,
                                      keras.layers.BatchNormalization)):
            print('Ignored', keras_layer.name)

        else:
            print('Not handled layer %s of type %s' % (keras_layer.name, type(keras_layer)))

    if output_classes is not None:
        grapher.layers[-1].unit_names = output_classes


def keras_load_cifar_test_data(test_data):
    """ Load CIFAR using Keras, return a sample of the test """

    (x_train, yTrain), (x_test, y_test) = keras.datasets.cifar10.load_data()

    test_data.set(x_test, y_test.ravel(),
        ['red', 'green', 'blue'],
        ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'])


def keras_load_mnist_test_data(test_data):
    """ Load MNIST using Keras, return a sample of the test """

    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    test_data.set(x_test, y_test, ['bw'], [str(d) for d in range(10)])


class KerasActivationMapper(AbstractActivationMapper):

    def __init__(self, model):
        AbstractActivationMapper.__init__(self)
        self.model = model

    def get_activation(self, img, layer: AbstractLayer, unit):

        # Format input if needed
        if img.dtype == np.uint8:
            img = img.astype(np.float) / 255

        # Create partial model
        intermediate_model = keras.models.Model(inputs=self.model.input,
                                                outputs=self.model.get_layer(layer.name).output)

        maps = intermediate_model.predict(np.expand_dims(img, 0))[0]
        if unit is None:
            return maps
        else:
            return maps[:, :, unit]
