from ..Grapher import Grapher


class AbstractModelSequence:
    """ Base class for a model sequence over training epochs """

    def __init__(self):
        self.number_epochs = 0
        self.current_epoch_index = -1
        self.title = ''

    def reset(self):
        self.number_epochs = 0
        self.current_epoch_index = -1
        self.title = ''

    # @abstract
    def list_models(self, directories, model_sequence_pattern='{model}_{epoch}'):
        """ List all models in directories """
        return []

    # @abstract
    def format_test_data(self):
        """ Prepare the test data to fit in the model (or raise ModelError) """
        return

    # @abstract
    def get_input_geometry(self):
        """ Return the type (as numpy dtype) and shape of the model input """
        return None, None

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
