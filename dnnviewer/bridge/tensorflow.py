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
    loss = keras_get_layer_attribute(model0, 'loss')
    if loss is not None:
        grapher.training_props['loss'] = get_caption(loss, _loss_captions)
    optimizer = keras_get_layer_attribute(model0, 'optimizer')
    if optimizer is not None:
        grapher.training_props['optimizer'] = get_caption(optimizer, _optimizer_captions)
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
            # Expand data to add a dimension of width 1
            data = np.expand_dims(data, len(data.shape))
        elif len(data.shape) == len(in_shape) + 1 and data.shape[-1] == 1:
            # Remove the last dimension of width 1
            data = np.squeeze(data, len(data.shape) - 1)
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
    if hasattr(model, 'loss_functions') and model.loss_functions is not None and len(model.loss_functions) > 0:
        # As in test model "LeNetLarge"
        loss = model.loss_functions[0]
    elif hasattr(model, 'loss') and model.loss is not None:
        # At least from Tensorflow 2.2
        loss = model.loss

    if loss is not None and (isinstance(loss, str) and loss == 'categorical_crossentropy') \
            or type(loss).__name__ == 'CategoricalCrossentropy':
        labels = keras.utils.to_categorical(labels)
    return labels


def keras_get_layer_attribute(keras_layer, attribute: str, sub_attr=None, look_in_config=False):
    """
    Safely get the value of an attribute from a Keras layer
    - If sub_attr is set, will return a dictionary with the listed sub attributes
    - If look_in_metadata is set, will also look in the layer metadata
    """
    value = None
    ret_attr = {}
    try:
        value = getattr(keras_layer, attribute)
    except AttributeError:
        pass

    if value is not None:
        if sub_attr is not None:

            # Return a dictionary with only the required sub-attributes
            for sa in sub_attr:
                try:
                    ret_attr[sa] = getattr(value, sa)
                except AttributeError:
                    pass
            return ret_attr

    # Could not find the attribute
    # For some reason, saved models sometimes are squeezing some attributes but keep them in the metadata
    if look_in_config:
        config = keras_layer.get_config()
        if config is not None and attribute in config.keys():
            if isinstance(config[attribute], dict) and 'config' in config[attribute].keys():
                value = config[attribute]['config']
            elif isinstance(config[attribute], (str, int, float)):
                value = config[attribute]

    if value is not None:
        if sub_attr is not None:

            # Return only requested sub-attributes as dictionary
            # Return a dictionary with only the required sub-attributes
            for sa in sub_attr:
                try:
                    ret_attr[sa] = value[sa]
                except AttributeError:
                    pass
            return ret_attr

    return value

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
