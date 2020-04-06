from ..layers.AbstractLayer import AbstractLayer
from ..Grapher import Grapher


class AbstractModelSequence:
    """ Base class for a model sequence over training epochs """

    def __init__(self):
        self.number_epochs = 0
        self.current_epoch_index = 0

    def first_epoch(self, grapher: Grapher):
        """ Go to first epoch in sequence and update graph """
        if self.number_epochs <= 0:
            print('No model to load')

        return self._load_model(grapher, 0)

    def last_epoch(self, grapher: Grapher):
        """ Go to last epoch in sequence and update graph """
        if self.number_epochs <= 0:
            print('No model to load')

        return self._load_model(grapher, self.number_epochs - 1)

    def previous_epoch(self, grapher: Grapher):
        """ Go to previous epoch in sequence and update graph """
        if self.number_epochs <= 0:
            print('No model to load')

        if self.current_epoch_index > 0:
            self._load_model(grapher, self.current_epoch_index - 1)
        return

    def next_epoch(self, grapher: Grapher):
        """ Go to next epoch in sequence and update graph """
        if self.number_epochs <= 0:
            print('No model to load')

        if self.current_epoch_index < self.number_epochs - 1:
            self._load_model(grapher, self.current_epoch_index + 1)
        return

    # @abstract
    def get_activation(self, img, layer: AbstractLayer, unit):
        """ Return the activation of a single unit or a set or the full layer as Numpy ndarrays
            @param img ndarray containing the input image
            @param layer output of the sub-network to compute activation
            @param unit selected unit within the target layer
        """
        return []

    # @abstract
    def _load_model(self, grapher: Grapher, model_index: int):
        """" model load to implement """
        return
