#
# Method to create graph view of a TensorFlow model
#

from ..Grapher import Grapher
from . import ModelError

import numpy as np
from tensorflow import keras
import logging

_logger = logging.getLogger(__name__)


def keras_set_model_properties(grapher: Grapher, model0: keras.models.Model):
    """ Map model level properties on grapher object """

    grapher.name = model0.name
    # Training properties
    grapher.training_props['loss'] = get_caption(model0.loss, _loss_captions)
    grapher.training_props['optimizer'] = get_caption(model0.optimizer, _optimizer_captions)
    return


def keras_prepare_input(in_type, in_shape, data):
    """ Expand dimensions and pad data if needed for the model
    Expect a batch of data that match the input of the model:
    - In case the first layer is a convolution, expects input to be 4D, adjust padding
    """

    # Format input if needed
    if data.dtype != in_type:
        data = data.astype(in_type) / 255

    # Handle input shape differences
    if np.array_equal(data.shape[1:], in_shape[1:]):
        return data

    # Dimension expansion
    if len(data.shape) != len(in_shape):
        if len(data.shape) == len(in_shape) - 1 and in_shape[-1] == 1:
            data = np.expand_dims(data, len(data.shape))
        else:
            raise ModelError(f"Incompatible test data sample shape, expected {in_shape}, test data shape {data.shape}")

    # Padding if required
    pad = False
    padding = np.zeros((len(data.shape), 2), dtype=np.int)
    for d in range(1, len(in_shape) - 1):
        delta = in_shape[d] - data.shape[d]
        if delta > 0:
            padding[d, 0] = delta // 2
            padding[d, 1] = delta - padding[d, 0]
            pad = True
    if pad:
        _logger.warning(f'Padding image with pad={padding}')
        data = np.pad(data, padding)

    return data


def keras_prepare_labels(model, labels):
    """ Preformat labels to be processed by the Keras model
      - mainly handle the case of categorical_crossentropy
    """
    loss = model.loss_functions[0]
    if (isinstance(loss, str) and loss == 'categorical_crossentropy') \
            or type(loss).__name__ == 'CategoricalCrossentropy':
        labels = keras.utils.to_categorical(labels)
    return labels


def get_caption(prop, dic):
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
