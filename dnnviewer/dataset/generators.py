from . import AbstractGenerator
import numpy as np


class RandomNormalGenerator(AbstractGenerator):

    def __init__(self, dtype, shape):
        AbstractGenerator.__init__(self, dtype, shape)
        self.rng = np.random.default_rng()

    def batch(self, length):
        cur_shape = self.shape.copy()
        cur_shape[0] = length
        self.current_sample = self.rng.standard_normal(size=cur_shape, dtype=self.dtype)
        return self.current_sample
