import numpy as np


def multi_to_one_dim(in_shape, in_index):
    """ Convert an index from a multi-dimension into the corresponding index in flat array """
    out_index = 0
    for dim, index in zip(in_shape, in_index):
        out_index = dim * out_index + index

    return out_index


def one_to_multi_dim(out_shape, in_index):
    """ Convert index in a flat array to index in corresponding multi-dimension array """
    n_dim = len(out_shape)
    out_index = np.empty(n_dim)
    remainder = in_index
    for i, dim in enumerate(reversed(out_shape)):
        index = remainder % dim
        remainder = remainder // dim
        out_index[n_dim - i - 1] = index

    return out_index


def multi_to_multi_dim(in_shape, out_shape, in_index):
    """" Multi dimension to multi dimension index transformation """
    return one_to_multi_dim(out_shape, multi_to_one_dim(in_shape, in_index))
