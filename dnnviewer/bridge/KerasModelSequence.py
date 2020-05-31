from . import tensorflow as tf_bridge
from .AbstractModelSequence import AbstractModelSequence, ModelError
from .AbstractActivationMapper import AbstractActivationMapper
from ..Grapher import Grapher
from ..layers.AbstractLayer import AbstractLayer

import tensorflow.keras as keras
import numpy as np
from pathlib import Path
import glob
import re
import logging
import traceback


class KerasModelSequence(AbstractModelSequence, AbstractActivationMapper):
    """ Handling a sequence of Keras models, saved as checkpoints or HDF5 """

    def __init__(self, test_data):
        AbstractModelSequence.__init__(self)
        self.model_paths = []
        self.current_model = None
        self.test_data = test_data

    # @override
    def reset(self):
        AbstractModelSequence.reset(self)
        self.model_paths = []
        self.current_model = None

    def load_single(self, file_path):
        """" Load a single Keras model from file_path"""

        self.reset()
        self.number_epochs = 1
        self.model_paths = [file_path]

    def load_sequence(self, dir_path):
        """ Load a sequence of models over epochs from dir_path with pattern on {epoch} tag """

        self.reset()
        checkpoint_glob = dir_path.replace('{epoch}', '[0-9]*')

        checkpoint_path_list = glob.glob(checkpoint_glob)
        checkpoint_epoch_regexp = re.compile(dir_path.replace('{epoch}', '([0-9]*)'))
        checkpoints = {int(checkpoint_epoch_regexp.search(path).group(1)): path for path in checkpoint_path_list}
        checkpoint_epochs = list(checkpoints)
        checkpoint_epochs.sort()
        self.model_paths = [checkpoints[i] for i in checkpoint_epochs]
        self.number_epochs = len(self.model_paths)

    # @override
    def list_models(self, directories, model_sequence_pattern='{model}_{epoch}'):
        """ List all models in directories """
        seq_pat1 = model_sequence_pattern.replace('{model}', '*').replace('{epoch}', '[0-9]*')
        seq_pat2 = model_sequence_pattern.replace('{model}', r'(\w+)').replace('{epoch}', '([0-9]+)')
        models = []

        logger = logging.getLogger(__name__)

        try:
            for path in directories:

                dir_path = Path(path)

                # HDF5 & TF files
                model_glob_hdf5 = str(dir_path / '*.h5')
                model_path_list = glob.glob(model_glob_hdf5)
                models.extend(model_path_list)
                model_glob_tf = str(dir_path / '*.tf')
                model_path_list = glob.glob(model_glob_tf)
                models.extend(model_path_list)

                # Checkpoints using pattern
                model_glob_seq = str(dir_path / seq_pat1)
                model_path_list = glob.glob(model_glob_seq)
                # Detect unique models
                reg2 = re.compile(seq_pat2)
                seq_model_path_list = [reg2.search(path).group(1) for path in model_path_list]
                model_path_list = [str(dir_path / model_sequence_pattern.replace('{model}', m))
                                   for m in set(seq_model_path_list)]
                models.extend(model_path_list)
        except Exception as e:
            logger.warning('Failed to list directories')
            logger.debug(traceback.format_exc(e))
        models.sort()
        return models

    # @override
    def get_activation(self, img, layer: AbstractLayer, unit=None):

        batch = tf_bridge.keras_prepare_input(self.current_model, np.expand_dims(img, 0))

        # Create partial model
        intermediate_model = keras.models.Model(inputs=self.current_model.input,
                                                outputs=self.current_model.get_layer(layer.name).output)

        # Expand dimension to create a mini-batch of 1 element
        maps = intermediate_model.predict(batch)[0]
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
        tf_bridge.keras_set_model_properties(grapher, self.current_model)

        # Create all other layers from the Keras Sequential model
        extractor = tf_bridge.NetworkExtractor(grapher, self.current_model, self.test_data)
        extractor.process()

        self.current_epoch_index = model_index
        return self.current_epoch_index
