# -*- coding: utf-8 -*-
# Authors: swolf <swolfforever@gmail.com>
# Date: 2022/11/11
# License: MIT License
"""Basic methods related with fourier transform."""
__all__ = [
    "zcrop",
    "fft1c",
    "ifft1c",
    "fft2c",
    "ifft2c",
    "fft3c",
    "ifft3c",
    "fftnc",
    "ifftnc",
    "fft1c_mod",
    "ifft1c_mod",
    "fft2c_mod",
    "ifft2c_mod",
    "fft3c_mod",
    "ifft3c_mod",
    "fftnc_mod",
    "ifftnc_mod",
    "fftmodn",
    "ifftmodn",
]


import numpy as np
from scipy.fft import fft, ifft, fft2, ifft2, fftn, ifftn, fftshift, ifftshift


def fft1c(X, axis=-1, workers=-1):
    """1d centered FFT.

    Parameters
    ----------
    X : numpy.ndarray
        Input array, can be complex.
    axis : int, optional
        Axis over which to compute the FFT. Defaults to -1 (the last axis).
    workers : int, optional
        Number of workers to use for parallel computation. Defaults to -1 (all available CPUs).
    Returns
    -------
    K : array_like
        K-space data.
    """
    K = ifftshift(X, axes=axis)
    K = fft(K, axis=axis, norm="ortho", workers=workers)
    K = fftshift(K, axes=axis)
    return K


def ifft1c(K, axis=-1, workers=-1):
    """1d centered iFFT.

    Parameters
    ----------
    K : numpy.ndarray
        Input K-space data, can be complex.
    axis : int, optional
        Axis over which to compute the iFFT. Defaults to -1 (the last axis).
    workers : int, optional
        Number of workers to use for parallel computation. Defaults to -1 (all available CPUs).
    Returns
    -------
    X : array_like
        Original data.
    """
    X = ifftshift(K, axes=axis)
    X = ifft(X, axis=axis, norm="ortho", workers=workers)
    X = fftshift(X, axes=axis)
    return X


def fft2c(X, axes=(-2, -1), workers=-1):
    """2d centered FFT.

    Parameters
    ----------
    X : numpy.ndarray
        Input array, can be complex.
    axes : tuple of int, optional
        Axes over which to compute the FFT. Defaults to (-2, -1) (the last two axes).
    workers : int, optional
        Number of workers to use for parallel computation. Defaults to -1 (all available CPUs).
    Returns
    -------
    K : array_like
        K-space data.
    """
    K = ifftshift(X, axes=axes)
    K = fft2(K, axes=axes, norm="ortho", workers=workers)
    K = fftshift(K, axes=axes)
    return K


def ifft2c(K, axes=(-2, -1), workers=-1):
    """2d centered iFFT.

    Parameters
    ----------
    K : numpy.ndarray
        Input K-space data, can be complex.
    axes : tuple of int, optional
        Axes over which to compute the iFFT. Defaults to (-2, -1) (the last two axes).
    workers : int, optional
        Number of workers to use for parallel computation. Defaults to -1 (all available CPUs).
    Returns
    -------
    X : array_like
        Original data.
    """
    X = ifftshift(K, axes=axes)
    X = ifft2(X, axes=axes, norm="ortho", workers=workers)
    X = fftshift(X, axes=axes)
    return X


def fft3c(X, axes=(-3, -2, -1), workers=-1):
    """3d centered FFT.

    Parameters
    ----------
    X : numpy.ndarray
        Input array, can be complex.
    axes : tuple of int, optional
        Axes over which to compute the FFT. Defaults to (-3, -2, -1) (the last three axes).
    workers : int, optional
        Number of workers to use for parallel computation. Defaults to -1 (all available CPUs).
    Returns
    -------
    K : array_like
        K-space data.
    """
    K = ifftshift(X, axes=axes)
    K = fftn(K, axes=axes, norm="ortho", workers=workers)
    K = fftshift(K, axes=axes)
    return K


def ifft3c(K, axes=(-3, -2, -1), workers=-1):
    """3d centered iFFT.

    Parameters
    ----------
    K : numpy.ndarray
        Input K-space data, can be complex.
    axes : tuple of int, optional
        Axes over which to compute the iFFT. Defaults to (-3, -2, -1) (the last three axes).
    workers : int, optional
        Number of workers to use for parallel computation. Defaults to -1 (all available CPUs).
    Returns
    -------
    X : array_like
        Original data.
    """
    X = ifftshift(K, axes=axes)
    X = ifftn(X, axes=axes, norm="ortho", workers=workers)
    X = fftshift(X, axes=axes)
    return X


def fftnc(X, axes=None, workers=-1):
    """Nd centered FFT.

    Parameters
    ----------
    X : numpy.ndarray
        Input array, can be complex.
    axes : tuple of int, optional
        Axes over which to compute the FFT. If not given, all axes are used.

    Returns
    -------
    K : array_like
        K-space data.
    """
    K = ifftshift(X, axes=axes)
    K = fftn(K, axes=axes, norm="ortho", workers=workers)
    K = fftshift(K, axes=axes)

    return K


def ifftnc(K, axes=None, workers=-1):
    """Nd centered iFFT.

    Parameters
    ----------
    K : numpy.ndarray
        Input K-space data, can be complex.
    axes : tuple of int, optional
        Axes over which to compute the iFFT. If not given, all axes are used.

    Returns
    -------
    X : array_like
        Original data.
    """
    X = ifftshift(K, axes=axes)
    X = ifftn(X, axes=axes, norm="ortho", workers=workers)
    X = fftshift(X, axes=axes)
    return X


def zcrop(X, shape):
    """Pad or crop a ND array around its center.

    The center is defined as floor(N/2) if N is odd, or N/2 if N is even.

    Parameters
    ----------
    X : numpy.ndarray
        Input array.
    shape : tuple of int
        Desired shape for output.

    Returns
    -------
    X_dest : array_like
        Cropped or padded array for shape.
    """
    if X.ndim != len(shape):
        raise ValueError(
            f"Input array has {X.ndim} dimensions, but shape has {len(shape)} dimensions."
        )

    X_dest = np.zeros_like(X, shape=shape)

    s_slices, d_slices = [], []
    for i, (s, d) in enumerate(zip(X.shape, shape)):
        if s < d:
            s_slices.append(slice(0, s))
            d_slices.append(slice(d // 2 - s // 2, d // 2 - s // 2 + s))
        else:
            s_slices.append(slice(s // 2 - d // 2, s // 2 - d // 2 + d))
            d_slices.append(slice(0, d))
    X_dest[tuple(d_slices)] = X[tuple(s_slices)]
    return X_dest


def fftmodn(X, axes=None):
    """Calculate the phase modulation array for centered FFT.

    This function computes the phase modulation array that can be used to replace
    `fftshift` and `ifftshift` operations for centered FFTs.

    Parameters
    ----------
    X : numpy.ndarray
        Input array, can be complex. Its shape is used to determine the modulation.
    axes : tuple of int, optional
        Axes over which to compute the FFT. If not given, all axes are used.

    Returns
    -------
    phase_mod : array_like
        phase modulation array.
    """
    if axes is None:
        axes = tuple(range(X.ndim))
    dtype = np.result_type(X.dtype, np.complex64)
    phase_mod = 1
    for axis in axes:
        N = X.shape[axis]
        S = N // 2
        mod_shape = [1] * X.ndim
        mod_shape[axis] = N
        phase_mod_axis = (
            np.exp(1j * 2 * np.pi * (np.arange(N) - S / 2) * S / N)
            .astype(dtype)
            .reshape(mod_shape)
        )
        phase_mod = (
            phase_mod * phase_mod_axis
        )  # utilize broadcasting to reduce the number of exps

    return phase_mod


def ifftmodn(X, axes=None):
    """Calculate the inverse phase modulation array for centered iFFT.

    This function computes the inverse phase modulation array that can be used to replace
    `fftshift` and `ifftshift` operations for centered iFFTs.

    Parameters
    ----------
    X : numpy.ndarray
        Input array, can be complex. Its shape is used to determine the modulation.
    axes : tuple of int, optional
        Axes over which to compute the FFT. If not given, all axes are used.

    Returns
    -------
    phase_mod : array_like
        phase modulation array.
    """
    if axes is None:
        axes = tuple(range(X.ndim))
    dtype = np.result_type(X.dtype, np.complex64)
    phase_mod = 1
    for axis in axes:
        N = X.shape[axis]
        S = N // 2
        mod_shape = [1] * X.ndim
        mod_shape[axis] = N
        phase_mod_axis = (
            np.exp(-1j * 2 * np.pi * (np.arange(N) - S / 2) * S / N)
            .astype(dtype)
            .reshape(mod_shape)
        )
        phase_mod = (
            phase_mod * phase_mod_axis
        )  # utilize broadcasting to reduce the number of exps

    return phase_mod


def fftnc_mod(X, axes=None, workers=-1, phase_mod=None):
    """Nd centered FFT with phase modulation.

    This function performs an N-dimensional centered Fast Fourier Transform
    using phase modulation instead of `fftshift`/`ifftshift`.

    Parameters
    ----------
    X : numpy.ndarray
        Input array, can be complex.
    axes : tuple of int, optional
        Axes over which to compute the FFT. If not given, all axes are used.
    workers : int, optional
        Number of workers to use for parallel computation. Defaults to -1 (all available CPUs).

    Returns
    -------
    K : array_like
        K-space data.
    """
    if phase_mod is None:
        phase_mod = fftmodn(X, axes=axes)
    K = X * phase_mod
    K = fftn(K, axes=axes, norm="ortho", workers=workers)
    K *= phase_mod
    return K


def ifftnc_mod(K, axes=None, workers=-1, phase_mod=None):
    """Nd centered iFFT with phase modulation.

    This function performs an N-dimensional centered Inverse Fast Fourier Transform
    using phase modulation instead of `fftshift`/`ifftshift`.

    Parameters
    ----------
    K : numpy.ndarray
        Input K-space data, can be complex.
    axes : tuple of int, optional
        Axes over which to compute the iFFT. If not given, all axes are used.
    workers : int, optional
        Number of workers to use for parallel computation. Defaults to -1 (all available CPUs).

    Returns
    -------
    X : array_like
        Original data.
    """
    if phase_mod is None:
        phase_mod = ifftmodn(K, axes=axes)
    X = K * phase_mod
    X = ifftn(X, axes=axes, norm="ortho", workers=workers)
    X *= phase_mod
    return X


def fft1c_mod(X, axis=-1, workers=-1, phase_mod=None):
    """1d centered FFT with phase modulation.

    This function performs a 1-dimensional centered Fast Fourier Transform
    using phase modulation instead of `fftshift`/`ifftshift`.

    Parameters
    ----------
    X : numpy.ndarray
        Input array, can be complex.
    axis : int, optional
        Axis over which to compute the FFT. Defaults to -1 (the last axis).
    workers : int, optional
        Number of workers to use for parallel computation. Defaults to -1 (all available CPUs).

    Returns
    -------
    K : array_like
        K-space data.
    """
    if phase_mod is None:
        phase_mod = fftmodn(X, axes=(axis,))
    K = X * phase_mod
    K = fft(K, axis=axis, norm="ortho", workers=workers)
    K *= phase_mod
    return K


def ifft1c_mod(K, axis=-1, workers=-1, phase_mod=None):
    """1d centered iFFT with phase modulation.

    This function performs a 1-dimensional centered Inverse Fast Fourier Transform
    using phase modulation instead of `fftshift`/`ifftshift`.

    Parameters
    ----------
    K : numpy.ndarray
        Input K-space data, can be complex.
    axis : int, optional
        Axis over which to compute the iFFT. Defaults to -1 (the last axis).
    workers : int, optional
        Number of workers to use for parallel computation. Defaults to -1 (all available CPUs).

    Returns
    -------
    X : array_like
        Original data.
    """
    if phase_mod is None:
        phase_mod = ifftmodn(K, axes=(axis,))
    X = K * phase_mod
    X = ifft(X, axis=axis, norm="ortho", workers=workers)
    X *= phase_mod
    return X


def fft2c_mod(X, axes=(-2, -1), workers=-1, phase_mod=None):
    """2d centered FFT with phase modulation.

    This function performs a 2-dimensional centered Fast Fourier Transform
    using phase modulation instead of `fftshift`/`ifftshift`.

    Parameters
    ----------
    X : numpy.ndarray
        Input array, can be complex.
    axes : tuple of int, optional
        Axes over which to compute the FFT. Defaults to (-2, -1) (the last two axes).
    workers : int, optional
        Number of workers to use for parallel computation. Defaults to -1 (all available CPUs).

    Returns
    -------
    K : array_like
        K-space data.
    """
    if phase_mod is None:
        phase_mod = fftmodn(X, axes=axes)
    K = X * phase_mod
    K = fft2(K, axes=axes, norm="ortho", workers=workers)
    K *= phase_mod
    return K


def ifft2c_mod(K, axes=(-2, -1), workers=-1, phase_mod=None):
    """2d centered iFFT with phase modulation.

    This function performs a 2-dimensional centered Inverse Fast Fourier Transform
    using phase modulation instead of `fftshift`/`ifftshift`.

    Parameters
    ----------
    K : numpy.ndarray
        Input K-space data, can be complex.
    axes : tuple of int, optional
        Axes over which to compute the iFFT. Defaults to (-2, -1) (the last two axes).
    workers : int, optional
        Number of workers to use for parallel computation. Defaults to -1 (all available CPUs).

    Returns
    -------
    X : array_like
        Original data.
    """
    if phase_mod is None:
        phase_mod = ifftmodn(K, axes=axes)
    X = K * phase_mod
    X = ifft2(X, axes=axes, norm="ortho", workers=workers)
    X *= phase_mod
    return X


def fft3c_mod(X, axes=(-3, -2, -1), workers=-1, phase_mod=None):
    """3d centered FFT with phase modulation.

    This function performs a 3-dimensional centered Fast Fourier Transform
    using phase modulation instead of `fftshift`/`ifftshift`.

    Parameters
    ----------
    X : numpy.ndarray
        Input array, can be complex.
    axes : tuple of int, optional
        Axes over which to compute the FFT. Defaults to (-3, -2, -1) (the last three axes).
    workers : int, optional
        Number of workers to use for parallel computation. Defaults to -1 (all available CPUs).

    Returns
    -------
    K : array_like
        K-space data.
    """
    if phase_mod is None:
        phase_mod = fftmodn(X, axes=axes)
    K = X * phase_mod
    K = fftn(K, axes=axes, norm="ortho", workers=workers)
    K *= phase_mod
    return K


def ifft3c_mod(K, axes=(-3, -2, -1), workers=-1, phase_mod=None):
    """3d centered iFFT with phase modulation.

    This function performs a 3-dimensional centered Inverse Fast Fourier Transform
    using phase modulation instead of `fftshift`/`ifftshift`.

    Parameters
    ----------
    K : numpy.ndarray
        Input K-space data, can be complex.
    axes : tuple of int, optional
        Axes over which to compute the iFFT. Defaults to (-3, -2, -1) (the last three axes).
    workers : int, optional
        Number of workers to use for parallel computation. Defaults to -1 (all available CPUs).

    Returns
    -------
    X : array_like
        Original data.
    """
    if phase_mod is None:
        phase_mod = ifftmodn(K, axes=axes)
    X = K * phase_mod
    X = ifftn(X, axes=axes, norm="ortho", workers=workers)
    X *= phase_mod
    return X
