import numpy as np
from psygine.decoders.utils.fourier import zcrop, fft1c_fast, ifft1c_fast, fft1c, ifft1c
import pytest

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

@pytest.mark.parametrize("size", [1, 127, 128])
def test_fft1c(size):
    """
    Tests fft1c_fast with complex 1D signals of both even and odd lengths.
    """
    np.random.seed(42) 
    real_part = np.random.randn(size)
    imag_part = np.random.randn(size)
    input_data = real_part + 1j * imag_part

    expected = fft1c(input_data)
    actual = fft1c_fast(input_data)

    assert np.allclose(actual, expected, rtol=1e-5, atol=1e-8), \
        f"FFT mismatch failed for size {size} (Type: {'Even' if size % 2 == 0 else 'Odd'})"
    
    expected = ifft1c(input_data)
    actual = ifft1c_fast(input_data)

    assert np.allclose(actual, expected, rtol=1e-5, atol=1e-8), \
        f"iFFT mismatch failed for size {size} (Type: {'Even' if size % 2 == 0 else 'Odd'})"
    
