

class AbstractActivationMapper:

    # @abstract
    def get_activation(self, img, layer, unit=None):
        """ Return the activation of a single unit or a set or the full layer as Numpy ndarrays
            @param img ndarray containing the input image
            @param layer output of the sub-network to compute activation
            @param unit selected unit within the target layer
        """
        return []
