import numpy as np
from dnnviewer.utils import statistics


def test_get_dense_strongest_weights():
    weights = np.array([[1, 2], [2, 3], [3, 2]])

    strongest_idx, strongest = statistics.get_strongest(weights, 1)

    assert (strongest_idx.shape == (1, 2))
    assert ((strongest_idx == np.array([[2, 1]])).all())
    assert (strongest.shape == (1, 2))
    assert ((strongest == np.array([[3, 3]])).all())


def test_get_dense_strongest_weights2():
    weights = np.array([[1, 2, 1], [2, 3, 4], [3, 2, 5]])

    strongest_idx, strongest = statistics.get_strongest(weights, 2)

    assert (strongest_idx.shape == (2, 3))
    assert ((strongest_idx == np.array([[1, 2, 1], [2, 1, 2]])).all())
    assert (strongest.shape == (2, 3))
    assert ((strongest == np.array([[2, 2, 4], [3, 3, 5]])).all())
