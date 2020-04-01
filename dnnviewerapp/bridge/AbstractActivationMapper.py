from ..layers.AbstractLayer import AbstractLayer


class AbstractActivationMapper:
    """ Abstract interface to get Activation maps """

    def get_activation(self, img, layer: AbstractLayer, unit):
        """ Return the activation of a single unit or a set or the full layer as Numpy ndarrays
            @param img ndarray containing the input image
        """
        return []
