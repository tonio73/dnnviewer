from dnnviewer.utils import array

import numpy as np


def test_multi_dim_to_one_dim():

    in_shape = [10, 17]
    in_index = [3, 5]

    out_index = array.multi_to_one_dim(in_shape, in_index)

    assert out_index == (3 * 17 + 5)


def test_multi_dim_to_one_dim_2():

    in_shape = [11, 10, 17]
    in_index = [3, 4, 5]

    out_index = array.multi_to_one_dim(in_shape, in_index)

    assert out_index == (5 + 17 * (4 + 10 * 3))


def test_one_dim_to_multi_dim():

    in_index = (5 + 17 * (4 + 10 * 3))
    out_shape = [11, 10, 17]

    out_index = array.one_to_multi_dim(out_shape, in_index)

    assert np.array_equal(out_index, [3, 4, 5])


def test_multi_to_multi_dim():

    in_shape = [32, 10]
    out_shape = [2, 5, 32]

    in_index = [10, 3]

    out_index = array.multi_to_multi_dim(in_shape, out_shape, in_index)

    assert np.array_equal(out_index, [0, 3, 7])
