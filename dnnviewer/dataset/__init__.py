import numpy as np


class AbstractGenerator:
    """ Abstract data generator """

    # @abstract
    def __init__(self, dtype, shape):
        self.dtype = dtype
        self.shape = shape
        self.current_sample = None

    def batch(self, length):
        """ Generate a batch of shape, return numpy array """
        cur_shape = self.shape.copy()
        cur_shape[0] = length
        self.current_sample = np.zeros(cur_shape, dtype=self.dtype)
        return self.current_sample
