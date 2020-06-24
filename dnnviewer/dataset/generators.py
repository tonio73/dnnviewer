import numpy as np

UNDEFINED = 'undefined'
RANDOM_NORMAL = 'random_normal'
ZEROS = 'zeros'
ONES = 'ones'

available_generators = {RANDOM_NORMAL: 'Random normal (Gauss with mean 0 and variance 1.0)',
                        ZEROS: 'Zeros',
                        ONES: 'Ones'}


class AbstractGenerator:
    """ Abstract data generator """

    # @abstract
    def __init__(self, id_, dtype, shape):
        self.id = id_
        self.dtype = dtype
        self.shape = shape
        self.current_sample = None

    def batch(self, length):
        """ Generate a batch of shape, return numpy array """
        cur_shape = self.shape.copy()
        cur_shape[0] = length
        self.current_sample = np.zeros(cur_shape, dtype=self.dtype)
        return self.current_sample

    def label(self):
        if self.id is not UNDEFINED:
            return available_generators[self.id]
        return ''


class RandomNormalGenerator(AbstractGenerator):

    def __init__(self, dtype, shape):
        AbstractGenerator.__init__(self, RANDOM_NORMAL, dtype, shape)
        self.rng = np.random.default_rng()

    def batch(self, length):
        cur_shape = self.shape.copy()
        cur_shape[0] = length
        self.current_sample = self.rng.standard_normal(size=cur_shape, dtype=self.dtype)
        return self.current_sample


class ZerosGenerator(AbstractGenerator):

    def __init__(self, dtype, shape):
        AbstractGenerator.__init__(self, ZEROS, dtype, shape)

    def batch(self, length):
        cur_shape = self.shape.copy()
        cur_shape[0] = length
        self.current_sample = np.zeros(shape=cur_shape, dtype=self.dtype)
        return self.current_sample


class OnesGenerator(AbstractGenerator):

    def __init__(self, dtype, shape):
        AbstractGenerator.__init__(self, ZEROS, dtype, shape)

    def batch(self, length):
        cur_shape = self.shape.copy()
        cur_shape[0] = length
        self.current_sample = np.ones(shape=cur_shape, dtype=self.dtype)
        return self.current_sample


builders = {
    RANDOM_NORMAL: RandomNormalGenerator,
    ZEROS: ZerosGenerator,
    ONES: OnesGenerator
}


def get_generators(generator_id):
    """ Factory for the generators """

    if generator_id in builders.keys():
        return builders[generator_id]

    return None
