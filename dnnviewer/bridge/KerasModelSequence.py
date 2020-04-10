from . import tensorflow
from .AbstractModelSequence import AbstractModelSequence, ModelError
from .AbstractActivationMapper import AbstractActivationMapper
from ..Grapher import Grapher
from ..layers.AbstractLayer import AbstractLayer

import tensorflow.keras as keras
import numpy as np
from pathlib import Path
import glob
import re


class KerasModelSequence(AbstractModelSequence, AbstractActivationMapper):
    """ Handling a sequence of Keras models, saved as checkpoints or HDF5 """

    def __init__(self, test_data):
        AbstractModelSequence.__init__(self)
        self.model_paths = []
        self.current_model = None
        self.test_data = test_data

    def load_single(self, file_path):
        """" Load a single Keras model from file_path"""
        self.number_epochs = 1
        self.model_paths = [file_path]

    def load_sequence(self, dir_path):
        """ Load a sequence of models over epochs from dir_path with pattern on {epoch} tag """
        checkpoint_glob = dir_path.replace('{epoch}', '[0-9]*')

        checkpoint_path_list = glob.glob(checkpoint_glob)
        checkpoint_epoch_regexp = re.compile(dir_path.replace('{epoch}', '([0-9]*)'))
        checkpoints = {int(checkpoint_epoch_regexp.search(path).group(1)): path for path in checkpoint_path_list}
        checkpoint_epochs = list(checkpoints)
        checkpoint_epochs.sort()
        self.model_paths = [checkpoints[i] for i in checkpoint_epochs]
        self.number_epochs = len(self.model_paths)

    # @override
    def get_activation(self, img, layer: AbstractLayer, unit=None):

        # Format input if needed
        if img.dtype == np.uint8:
            img = img.astype(np.float) / 255

        # Handle convolution input
        if len(self.current_model.layers) > 0 \
                and type(self.current_model.layers[0]).__name__ == 'Conv2D':

            # Padding if required
            pad = False
            padding = np.zeros((2, 2), dtype=np.int)
            input_shape = self.current_model.layers[0].input.get_shape()
            for d in [0, 1]:
                delta = input_shape[d + 1] - img.shape[0]
                if delta > 0:
                    padding[d, 1] = delta
                    pad = True
            if pad:
                img = np.pad(img, padding)

            # Convolution requires 3D input
            if len(img.shape) == 2:
                img = np.expand_dims(img, 2)

        # Create partial model
        intermediate_model = keras.models.Model(inputs=self.current_model.input,
                                                outputs=self.current_model.get_layer(layer.name).output)

        maps = intermediate_model.predict(np.expand_dims(img, 0))[0]
        if unit is None:
            return maps
        else:
            return maps[:, :, unit]

    def _load_model(self, grapher: Grapher, model_index: int):

        model_path = Path(self.model_paths[model_index])

        if not model_path.exists():
            raise ModelError('Model path not found %s' % str(model_path))

        self.current_model = keras.models.load_model(str(model_path))

        # Top level properties of the DNN model
        tensorflow.keras_set_model_properties(grapher, self.current_model)

        # Create all other layers from the Keras Sequential model
        tensorflow.keras_extract_sequential_network(grapher, self.current_model, self.test_data)

        self.current_epoch_index = model_index
        return self.current_epoch_index
