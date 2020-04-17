#
# Method to create graph view of a TensorFlow model
#

from .AbstractModelSequence import ModelError
from ..Grapher import Grapher
from ..layers.Input import Input
from ..layers.Dense import Dense
from ..layers.Convo2D import Convo2D
from ..TestData import TestData

import numpy as np
from tensorflow import keras
import tensorflow as tf
import logging


def keras_set_model_properties(grapher: Grapher, model0: keras.models.Model):
    """ Map model level properties on grapher object """

    grapher.name = model0.name
    # Training properties
    grapher.training_props['loss'] = _get_caption(model0.loss, _loss_captions)
    grapher.training_props['optimizer'] = _get_caption(model0.optimizer, _optimizer_captions)
    return


def keras_extract_sequential_network(grapher: Grapher, model: keras.models.Model, test_data: TestData):
    """ Create a graphical representation of a Keras Sequential model's layers
        Compute gradients from test samples
    """

    logger = logging.getLogger(__name__)

    if not isinstance(model, keras.models.Sequential):
        raise ModelError("Unexpected model of type '%s', expecting Sequential model" % type(model))

    if len(model.layers) == 0:
        logger.error("Empty model")
        return

    theme = grapher.theme

    previous_layer = None
    grapher.clear_layers()

    # Input placeholder if first layer is a layer with weights
    if len(model.layers[0].get_weights()) > 0:
        input_dim = model.layers[0].get_weights()[0].shape[-2]
        input_layer = Input('input', input_dim, theme, test_data.input_classes)
        grapher.add_layer(input_layer)
        previous_layer = input_layer

    # Compute gradients applying a mini-batch of test data
    try:
        if test_data.has_test_sample:
            with tf.GradientTape() as tape:
                n_grad_samples = 256
                y_est = model(keras_prepare_input(model, test_data.x[:n_grad_samples]))
                objective = model.loss_functions[0](keras_prepare_labels(model, test_data.y[:n_grad_samples]),
                                                    y_est)
                grads = tape.gradient(objective, model.trainable_variables)
        else:
            grads = None
    except Exception:
        logger.error('Unable to compute gradients for model %s', model.name)
        grads = None

    idx_grads = 0
    for keras_layer in model.layers:

        layer_class = type(keras_layer).__name__

        if layer_class == 'Dense':
            layer = Dense(keras_layer.name, keras_layer.output_shape[-1],
                          keras_layer.weights[0].numpy(), grads[idx_grads].numpy() if grads else None,
                          theme)
            grapher.add_layer(layer)
            previous_layer = layer

        elif layer_class == 'Conv2D':
            layer = Convo2D(keras_layer.name, keras_layer.output_shape[-1],
                            keras_layer.weights[0].numpy(), grads[idx_grads].numpy() if grads else None,
                            theme)
            grapher.add_layer(layer)
            previous_layer = layer

        elif layer_class == 'Flatten':
            if isinstance(previous_layer, Convo2D):
                previous_layer.flatten_output = True
            elif previous_layer is None:
                # Input layer
                input_dim = keras_layer.get_output_shape_at(0)[-1]
                input_layer = Input('input', input_dim, theme, test_data.input_classes)
                grapher.add_layer(input_layer)
                previous_layer = input_layer

        elif layer_class in _keras_ignored_layers:
            logger.info('Ignored %s', keras_layer.name)

        else:
            logger.error('Not handled layer %s of type %s' % (keras_layer.name, type(keras_layer)))

        idx_grads += len(keras_layer.trainable_weights)

    if test_data.output_classes is not None:
        grapher.layers[-1].unit_names = test_data.output_classes


def keras_prepare_input(model, data):
    """ Expand dimensions and pad data if needed for the model
    Expect a batch of data that match the input of the model:
    - In case the first layer is a convolution, expects input to be 4D, adjust padding
    """

    logger = logging.getLogger(__name__)

    # Format input if needed
    if data.dtype == np.uint8:
        data = data.astype(np.float) / 255

    # Handle convolution input
    if len(model.layers) > 0 and type(model.layers[0]).__name__ == 'Conv2D':

        # Convolution requires 3D input
        if len(data.shape) == 3:
            data = np.expand_dims(data, 3)

        # Padding if required
        pad = False
        padding = np.zeros((len(data.shape), 2), dtype=np.int)
        input_shape = model.layers[0].input.get_shape()
        for d in [1, 2]:
            delta = input_shape[d] - data.shape[d]
            if delta > 0:
                padding[d, 0] = delta // 2
                padding[d, 1] = delta - padding[d, 0]
                pad = True
        if pad:
            logger.warning(f'Padding image before activation with pad={padding}')
            data = np.pad(data, padding)

    return data


def keras_prepare_labels(model, labels):
    """ Preformat labels, mainly handle the case of categorical_crossentropy """
    loss = model.loss_functions[0]
    if (isinstance(loss, str) and loss == 'categorical_crossentropy') \
            or type(loss).__name__ == 'CategoricalCrossentropy':
        labels = keras.utils.to_categorical(labels)
    return labels


def _get_caption(prop, dic):
    """ Get a textual caption of a keras property that my be $
        - a literal string label (e.g. 'mse')
        - an instance of a class
        - a reference to a function
    """
    if prop is not None:
        if isinstance(prop, str):
            prop_extract = prop.lower().replace('_', '')
            if prop_extract in dic:
                return dic[prop_extract]
            else:
                return prop
        else:
            prop_extract = type(prop).__name__.lower()
            if prop_extract == 'function':
                prop_extract = prop.__name__.lower().replace('_', '')
                if prop_extract in dic:
                    return dic[prop_extract]
                else:
                    return prop.__name__
            elif prop_extract in dic:
                return dic[prop_extract]
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
