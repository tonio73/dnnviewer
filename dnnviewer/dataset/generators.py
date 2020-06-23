from . import AbstractGenerator
import numpy as np

RANDOM_NORMAL = 'random_normal'


class RandomNormalGenerator(AbstractGenerator):

    def __init__(self, dtype, shape):
        AbstractGenerator.__init__(self, dtype, shape)
        self.rng = np.random.default_rng()

    def batch(self, length):
        cur_shape = self.shape.copy()
        cur_shape[0] = length
        self.current_sample = self.rng.standard_normal(size=cur_shape, dtype=self.dtype)
        return self.current_sample


builders = {
    RANDOM_NORMAL: RandomNormalGenerator
}


def get_generators(generator_id):
    """ Factory for the generators """

    if generator_id in builders.keys():
        return builders[generator_id]

    return None
