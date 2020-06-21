import numpy as np


class AbstractGenerator:
    """ Abstract data generator """

    # @abstract
    def __init__(self):
        self.current_sample = None

    def batch(self, shape):
        """ Generate a batch of shape, return numpy array """
        self.current_sample = np.zeros(shape)
        return self.current_sample
