import pytest
import numpy as np
from psygine.decoders.utils.fourier import *


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


@pytest.mark.parametrize(
    "shape", [(1,), (127,), (128,), (63, 64), (32, 33, 34), (31, 32, 33, 34)]
)
def test_fft_mods(shape):
    """
    Tests fftmodn with complex nD signals of both even and odd lengths.
    """
    np.random.seed(42)
    input_data = np.random.randn(*shape) + 1j * np.random.randn(*shape)

    axes = tuple(range(len(shape)))

    # Test specific dimension functions
    if len(shape) == 1:
        # Test fftmod1 vs fft1c
        expected_fft1 = fft1c(input_data)
        actual_fft1 = fftmod1(input_data)
        assert np.allclose(
            actual_fft1, expected_fft1, rtol=1e-5, atol=1e-8
        ), f"FFT1 mismatch failed for shape {shape}"

        # Test ifftmod1 vs ifft1c
        expected_ifft1 = ifft1c(input_data)
        actual_ifft1 = ifftmod1(input_data)
        assert np.allclose(
            actual_ifft1, expected_ifft1, rtol=1e-5, atol=1e-8
        ), f"iFFT1 mismatch failed for shape {shape}"
    elif len(shape) == 2:
        # Test fftmod2 vs fft2c
        expected_fft2 = fft2c(input_data, axes=(-2, -1))
        actual_fft2 = fftmod2(input_data)
        assert np.allclose(
            actual_fft2, expected_fft2, rtol=1e-5, atol=1e-8
        ), f"FFT2 mismatch failed for shape {shape}"

        # Test ifftmod2 vs ifft2c
        expected_ifft2 = ifft2c(input_data, axes=(-2, -1))
        actual_ifft2 = ifftmod2(input_data)
        assert np.allclose(
            actual_ifft2, expected_ifft2, rtol=1e-5, atol=1e-8
        ), f"iFFT2 mismatch failed for shape {shape}"
    elif len(shape) == 3:
        # Test fftmod3 vs fft3c
        expected_fft3 = fft3c(input_data, axes=(-3, -2, -1))
        actual_fft3 = fftmod3(input_data)
        assert np.allclose(
            actual_fft3, expected_fft3, rtol=1e-5, atol=1e-8
        ), f"FFT3 mismatch failed for shape {shape}"

        # Test ifftmod3 vs ifft3c
        expected_ifft3 = ifftnc(input_data, axes=(-3, -2, -1))
        actual_ifft3 = ifftmod3(input_data)
        assert np.allclose(
            actual_ifft3, expected_ifft3, rtol=1e-5, atol=1e-8
        ), f"iFFT3 mismatch failed for shape {shape}"
    else:
        # Test fftmodn vs fftnc
        expected_fft = fftnc(input_data, axes=axes)
        actual_fft = fftmodn(input_data, axes=axes)
        assert np.allclose(
            actual_fft, expected_fft, rtol=1e-5, atol=1e-8
        ), f"FFT mismatch failed for shape {shape}"

        # Test ifftmodn vs ifftnc
        expected_ifft = ifftnc(input_data, axes=axes)
        actual_ifft = ifftmodn(input_data, axes=axes)
        assert np.allclose(
            actual_ifft, expected_ifft, rtol=1e-5, atol=1e-8
        ), f"iFFT mismatch failed for shape {shape}"
