#
# Method to create graph view of a TensorFlow model
#

from tensorflow import keras

from .AbstractModelSequence import ModelError
from ..Grapher import Grapher
from ..layers.Input import Input
from ..layers.Dense import Dense
from ..layers.Convo2D import Convo2D
from ..SimpleColorScale import SimpleColorScale

import logging


def keras_set_model_properties(grapher: Grapher, model0: keras.models.Model):
    """ Map model level properties on grapher object """

    grapher.name = model0.name
    # Training properties
    grapher.training_props['loss'] = _get_caption(model0.loss, _loss_captions)
    grapher.training_props['optimizer'] = _get_caption(model0.optimizer, _optimizer_captions)
    return


def keras_extract_sequential_network(grapher: Grapher, model: keras.models.Model, test_data):
    """ Create a graphical representation of a Keras Sequential model's layers """

    logger = logging.getLogger(__name__)

    if not isinstance(model, keras.models.Sequential):
        raise ModelError("Unexpected model of type '%s', expecting Sequential model" % type(model))

    if len(model.layers) == 0:
        logger.error("Empty model")
        return

    plotly_theme = grapher.plotly_theme
    color_scale = SimpleColorScale()

    previous_layer = None
    grapher.clear_layers()

    # Input placeholder if first layer is a layer with weights
    if len(model.layers[0].get_weights()) > 0:
        input_dim = model.layers[0].get_weights()[0].shape[-2]
        if test_data.input_classes is not None and len(test_data.input_classes) != input_dim:
            logger.error("Wrong length of input classes, got %d, expecting %d" %
                         (len(test_data.input_classes), input_dim))

        input_layer = Input('input', input_dim, plotly_theme, test_data.input_classes)
        grapher.add_layer(input_layer)
        previous_layer = input_layer

    for keras_layer in model.layers:
        layer_class = type(keras_layer).__name__
        if layer_class == 'Dense':
            layer = Dense(keras_layer.name, keras_layer.output_shape[-1], keras_layer.weights[0].numpy(),
                          plotly_theme, color_scale)
            grapher.add_layer(layer)
            previous_layer = layer

        elif layer_class == 'Conv2D':
            layer = Convo2D(keras_layer.name, keras_layer.output_shape[-1], keras_layer.weights[0].numpy(),
                            plotly_theme, color_scale)
            grapher.add_layer(layer)
            previous_layer = layer

        elif layer_class == 'Flatten':
            if isinstance(previous_layer, Convo2D):
                previous_layer.flatten_output = True
            elif previous_layer is None:
                # Input layer
                input_dim = keras_layer.get_output_shape_at(0)[-1]
                input_layer = Input('input', input_dim, plotly_theme, test_data.input_classes)
                grapher.add_layer(input_layer)
                previous_layer = input_layer

        elif layer_class in _keras_ignored_layers:
            logger.info('Ignored', keras_layer.name)

        else:
            logger.error('Not handled layer %s of type %s' % (keras_layer.name, type(keras_layer)))

    if test_data.output_classes is not None:
        grapher.layers[-1].unit_names = test_data.output_classes


def keras_load_cifar_test_data(test_data):
    """ Load CIFAR using Keras, return a sample of the test """

    (_, _), (x_test, y_test) = keras.datasets.cifar10.load_data()

    test_data.set(x_test, y_test.ravel(),
                  ['red', 'green', 'blue'],
                  ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'])


def keras_load_mnist_test_data(test_data):
    """ Load MNIST using Keras, return a sample of the test """

    (_, _), (x_test, y_test) = keras.datasets.mnist.load_data()

    test_data.set(x_test, y_test,
                  ['bw'],
                  [str(d) for d in range(10)])


def keras_load_mnistfashion_test_data(test_data):
    """ Load Fashion MNIST using Keras, return a sample of the test """

    (_, _), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

    test_data.set(x_test, y_test,
                  ['bw'],
                  ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker',
                   'Bag', 'Ankle boot'])


def _get_caption(prop, dic):
    """ Get a textual caption of a keras property that my be $
        - a literal string label (e.g. 'mse')
        - an instance of a class
        - a reference to a function
    """
    if prop is not None:
        if isinstance(prop, str):
            loss_extract = prop.lower().replace('_', '')
            if loss_extract in dic:
                return dic[loss_extract]
            else:
                return prop
        else:
            loss_extract = type(prop).__name__.lower()
            if loss_extract == 'function':
                loss_extract = prop.__name__.lower().replace('_', '')
                if loss_extract in dic:
                    return dic[loss_extract]
                else:
                    return prop.__name__
            elif loss_extract in dic:
                return dic[loss_extract]
            else:
                return type(prop).__name__


_keras_ignored_layers = ['ActivityRegularization', 'Dropout',
                         'SpatialDropout1D', 'SpatialDropout2D',
                         'SpatialDropout3D',
                         'Activation',
                         'MaxPooling1D', 'MaxPooling2D', 'MaxPooling3D',
                         'AveragePooling1D', 'AveragePooling2D',
                         'AveragePooling3D',
                         'GlobalAveragePooling1D', 'GlobalAveragePooling2D',
                         'GlobalAveragePooling3D',
                         'GlobalMaxPooling1D', 'GlobalMaxPooling2D',
                         'GlobalMaxPooling3D',
                         'BatchNormalization']

_loss_captions = {'binarycrossentropy': 'Binary cross-entropy',
                  'categoricalcrossentropy': 'Categorical cross-entropy',
                  'categoricalhinge': 'Categorical hinge',
                  'cosinesimilarity': 'Cosine similarity',
                  'hinge': 'Hinge',
                  'huber': 'Huber',
                  'kld': 'Kullback-Leibler divergence',
                  'kullbackleiblerdivergence': 'Kullback-Leibler divergence',
                  'logcosh': 'Logarithm of the hyperbolic cosine of the prediction error',
                  'mae': 'Mean absolute error',
                  'mape': 'Mean absolute percentage error',
                  'meanabsoluteerror': 'Mean absolute error',
                  'meanabsolutepercentageerror': 'Mean absolute percentage error',
                  'meansquarederror': 'Mean squared error',
                  'meansquaredlogarithmicerror': 'Mean squared log error',
                  'mse': 'Mean squared error',
                  'msle': 'Mean squared log error',
                  'poisson': 'Poisson',
                  'sparsecategoricalcrossentropy': 'Sparse cross-entropy',
                  'squaredhinge': 'Squared hinge'}

_optimizer_captions = {'adadelta': 'Adadelta algorithm',
                       'adagrad': 'Adagrad algorithm',
                       'adam': 'Adam algorithm',
                       'adamax': 'Adamax algorithm',
                       'ftrl': 'FTRL algorithm',
                       'nadam': 'NAdam algorithm',
                       'rmsprop': 'RMSprop algorithm',
                       'sgd': 'Stochastic gradient descent and momentum'}
