from ..Grapher import Grapher


class AbstractModelSequence:
    """ Base class for a model sequence over training epochs """

    def __init__(self):
        self.number_epochs = 0
        self.current_epoch_index = -1

    def reset(self):
        self.number_epochs = 0
        self.current_epoch_index = -1

    # @abstract
    def list_models(self, directories, model_sequence_pattern='{model}_{epoch}'):
        """ List all models in directories """
        return []

    def first_epoch(self, grapher: Grapher):
        """ Go to first epoch in sequence and update graph """
        assert self.number_epochs > 0

        if self.current_epoch_index == 0:
            return self.current_epoch_index
        return self._load_model(grapher, 0)

    def last_epoch(self, grapher: Grapher):
        """ Go to last epoch in sequence and update graph """
        assert self.number_epochs > 0

        if self.current_epoch_index == self.number_epochs - 1:
            return self.current_epoch_index
        return self._load_model(grapher, self.number_epochs - 1)

    def previous_epoch(self, grapher: Grapher):
        """ Go to previous epoch in sequence and update graph """
        assert self.number_epochs > 0

        if self.current_epoch_index == 0:
            return self.current_epoch_index
        return self._load_model(grapher, self.current_epoch_index - 1)

    def next_epoch(self, grapher: Grapher):
        """ Go to next epoch in sequence and update graph """
        assert self.number_epochs > 0

        if self.current_epoch_index == self.number_epochs - 1:
            return self.current_epoch_index
        return self._load_model(grapher, self.current_epoch_index + 1)

    # @abstract
    def _load_model(self, grapher: Grapher, model_index: int):
        """" model load to implement
            @return current epoch index
        """
        return self.current_epoch_index


class ModelError(Exception):
    """ Exception to notify issues in the bridge package """
    def __init__(self, message):
        self.message = message
