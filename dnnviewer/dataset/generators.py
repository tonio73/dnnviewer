from . import AbstractGenerator
import numpy as np


class RandomNormalGenerator(AbstractGenerator):

    def batch(self, shape):
        self.current_sample = np.random.randn(*shape)
        return self.current_sample
