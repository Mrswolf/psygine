import numpy as np
from psygine.decoders.utils.fourier import zcrop


def test_zcrop():
    # Test 1D arrays
    X_1D_odd = np.array([1, 2, 3])
    X_1D_even = np.array([1, 2, 3, 4])
    Y = zcrop(X_1D_odd, (1,))
    assert np.array_equal(Y, np.array([2]))

    Y = zcrop(X_1D_odd, (2,))
    assert np.array_equal(Y, np.array([1, 2]))

    Y = zcrop(X_1D_even, (1,))
    assert np.array_equal(Y, np.array([3]))

    Y = zcrop(X_1D_even, (2,))
    assert np.array_equal(Y, np.array([2, 3]))

    Y = zcrop(X_1D_odd, (4,))
    assert np.array_equal(Y, np.array([0, 1, 2, 3]))

    Y = zcrop(X_1D_odd, (5,))
    assert np.array_equal(Y, np.array([0, 1, 2, 3, 0]))

    Y = zcrop(X_1D_even, (5,))
    assert np.array_equal(Y, np.array([1, 2, 3, 4, 0]))

    Y = zcrop(X_1D_even, (6,))
    assert np.array_equal(Y, np.array([0, 1, 2, 3, 4, 0]))

    # Test 2D arrays
    A_odd = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    A_even = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])

    Y = zcrop(A_odd, (2, 4))
    assert np.array_equal(Y, np.array([[0, 1, 2, 3], [0, 4, 5, 6]]))

    Y = zcrop(A_even, (4, 3))
    assert np.array_equal(Y, np.array([[0, 0, 0], [2, 3, 4], [6, 7, 8], [10, 11, 12]]))

    # Test 3D arrays
    A_3D = np.arange(24).reshape((2, 3, 4))
    Y = zcrop(A_3D, (3, 1, 2))
    assert np.array_equal(Y.flatten(order="A"), np.array([5, 6, 17, 18, 0, 0]))
