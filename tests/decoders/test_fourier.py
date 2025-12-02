import numpy as np
from psygine.decoders.utils.fourier import (
    zcrop, fft1c, ifft1c, fftnc, ifftnc,
    fftmod1, ifftmod1, fftmod2, ifftmod2,
    fftmod3, ifftmod3, fftmodn, ifftmodn
)
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
def test_fft1c_mod(size):
    """
    Tests fftmod1 with complex 1D signals of both even and odd lengths.
    """
    np.random.seed(42) 
    real_part = np.random.randn(size)
    imag_part = np.random.randn(size)
    input_data = real_part + 1j * imag_part

    expected = fft1c(input_data)
    actual = fftmod1(input_data)

    assert np.allclose(actual, expected, rtol=1e-5, atol=1e-8), (
        f"FFT mismatch failed for size {size} (Type: {'Even' if size % 2 == 0 else 'Odd'})"
    )
    
    expected = ifft1c(input_data)
    actual = ifftmod1(input_data)

    assert np.allclose(actual, expected, rtol=1e-5, atol=1e-8), (
        f"iFFT mismatch failed for size {size} (Type: {'Even' if size % 2 == 0 else 'Odd'})"
    )

@pytest.mark.parametrize("shape", [(128,), (127,), (64, 64), (63, 63), (32, 33, 34)])
def test_fft_mods(shape):
    """
    Tests fftmodn with complex nD signals of both even and odd lengths.
    """
    np.random.seed(42)
    input_data = np.random.randn(*shape) + 1j * np.random.randn(*shape)

    axes = tuple(range(len(shape)))

    # Test fftmodn vs fftnc
    expected_fft = fftnc(input_data, axes=axes)
    actual_fft = fftmodn(input_data, axes=axes)
    assert np.allclose(actual_fft, expected_fft, rtol=1e-5, atol=1e-8), (
        f"FFT mismatch failed for shape {shape}"
    )

    # Test ifftmodn vs ifftnc
    expected_ifft = ifftnc(input_data, axes=axes)
    actual_ifft = ifftmodn(input_data, axes=axes)
    assert np.allclose(actual_ifft, expected_ifft, rtol=1e-5, atol=1e-8), (
        f"iFFT mismatch failed for shape {shape}"
    )

    # Test specific dimension functions
    if len(shape) == 2:
        # Test fftmod2 vs fftnc
        expected_fft2 = fftnc(input_data, axes=(-2,-1))
        actual_fft2 = fftmod2(input_data)
        assert np.allclose(actual_fft2, expected_fft2, rtol=1e-5, atol=1e-8), (
            f"FFT2 mismatch failed for shape {shape}"
        )
        
        # Test ifftmod2 vs ifftnc
        expected_ifft2 = ifftnc(input_data, axes=(-2,-1))
        actual_ifft2 = ifftmod2(input_data)
        assert np.allclose(actual_ifft2, expected_ifft2, rtol=1e-5, atol=1e-8), (
            f"iFFT2 mismatch failed for shape {shape}"
        )

    if len(shape) == 3:
        # Test fftmod3 vs fftnc
        expected_fft3 = fftnc(input_data, axes=(-3,-2,-1))
        actual_fft3 = fftmod3(input_data)
        assert np.allclose(actual_fft3, expected_fft3, rtol=1e-5, atol=1e-8), (
            f"FFT3 mismatch failed for shape {shape}"
        )

        # Test ifftmod3 vs ifftnc
        expected_ifft3 = ifftnc(input_data, axes=(-3,-2,-1))
        actual_ifft3 = ifftmod3(input_data)
        assert np.allclose(actual_ifft3, expected_ifft3, rtol=1e-5, atol=1e-8), (
            f"iFFT3 mismatch failed for shape {shape}"
        )
