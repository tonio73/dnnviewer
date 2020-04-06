from ..Grapher import Grapher
from .AbstractModelSequence import AbstractModelSequence
from ..layers.AbstractLayer import AbstractLayer
from . import tensorflow

import tensorflow.keras as keras
import numpy as np
from pathlib import Path
import glob
import re


class KerasModelSequence(AbstractModelSequence):

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

    # @override
    def get_activation(self, img, layer: AbstractLayer, unit):

        # Format input if needed
        if img.dtype == np.uint8:
            img = img.astype(np.float) / 255

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
            print('Model path not found', str(model_path))
            return

        self.current_model = keras.models.load_model(str(model_path))

        # Top level properties of the DNN model
        tensorflow.keras_set_model_properties(grapher, self.current_model)

        # Create all other layers from the Keras Sequential model
        tensorflow.keras_extract_sequential_network(grapher, self.current_model, self.test_data)

        self.current_epoch_index = model_index