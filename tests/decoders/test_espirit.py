import pytest
import numpy as np
from psygine.decoders.mri import construct_hankel, sum_of_diags


def test_sum_of_diags():
    with pytest.raises(ValueError, match="A must be at least 2-dimensional."):
        sum_of_diags(np.array([1, 2, 3]))

    with pytest.raises(ValueError, match="The two dimensions of A must be equal."):
        sum_of_diags(np.array([[1, 2, 3], [4, 5, 6]]))

    # Test 2D array
    A = np.arange(9).reshape((3, 3))
    B = sum_of_diags(A)
    assert np.array_equal(B, np.array([6, 10, 12, 6, 2]))

    # Test 3D array
    A = np.arange(18).reshape((2, 3, 3))
    B = sum_of_diags(A, axis1=-2, axis2=-1)
    assert np.array_equal(B, np.array([[6, 10, 12, 6, 2], [15, 28, 39, 24, 11]]))

    B = sum_of_diags(A, axis1=-1, axis2=-2)
    assert np.array_equal(B, np.array([[2, 6, 12, 10, 6], [11, 24, 39, 28, 15]]))

    A = np.transpose(A, (1, 0, 2))
    B = sum_of_diags(A, axis1=0, axis2=2)
    assert np.array_equal(B, np.array([[6, 10, 12, 6, 2], [15, 28, 39, 24, 11]]))

    B = sum_of_diags(A, axis1=2, axis2=0)
    assert np.array_equal(B, np.array([[2, 6, 12, 10, 6], [11, 24, 39, 28, 15]]).T)
