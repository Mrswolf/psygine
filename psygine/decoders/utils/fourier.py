# -*- coding: utf-8 -*-
# Authors: swolf <swolfforever@gmail.com>
# Date: 2022/11/11
# License: MIT License
"""Basic methods related with fourier transform.
"""
import numpy as np
from scipy.fft import fft, ifft, fft2, ifft2, fftn, ifftn, fftshift, ifftshift

__all__ = [
    "fft1c",
    "ifft1c",
    "fft2c",
    "ifft2c",
    "fft3c",
    "ifft3c",
    "fftnc",
    "ifftnc",
    "zcrop",
]


def fft1c(X, axis=-1, workers=-1):
    r"""1d FFT.

    Parameters
    ----------
    X : array_like
        Input array, can be complex.
    axis : int, default -1
        Axis over which to compute the FFT. If not given, the last axis is used.

    Returns
    -------
    K : array_like
        K-space data.
    """
    K = fftshift(
        fft(ifftshift(X, axes=axis), axis=axis, norm="ortho", workers=workers), axes=axis
    )
    return K


def ifft1c(K, axis=-1, workers=-1):
    r"""1d iFFT.

    Parameters
    ----------
    K : array_like
        Input K space data, can be complex.
    axis : int, default -1
        Axis over which to compute the iFFT. If not given, the last axis is used.

    Returns
    -------
    X : array_like
        Original data.
    """
    X = fftshift(
        ifft(ifftshift(K, axes=axis), axis=axis, norm="ortho", workers=workers),
        axes=axis,
    )
    return X


def fft2c(X, axes=(-2, -1), workers=-1):
    r"""2d FFT.

    Parameters
    ----------
    X : array_like
        Input array, can be complex.
    axes : int, default (-2, -1)
        Axes over which to compute the FFT. If not given, the last two axes are used.

    Returns
    -------
    K : array_like
        K-space data.
    """
    K = fftshift(
        fft2(ifftshift(X, axes=axes), axes=axes, norm="ortho", workers=workers),
        axes=axes,
    )
    return K


def ifft2c(K, axes=(-2, -1), workers=-1):
    r"""2d iFFT.

    Parameters
    ----------
    K : array_like
        Input K space data, can be complex.
    axes : int, default (-2, -1)
        Axes over which to compute the iFFT. If not given, the last two axes are used.

    Returns
    -------
    X : array_like
        Original data.
    """
    X = fftshift(
        ifft2(ifftshift(K, axes=axes), axes=axes, norm="ortho", workers=workers),
        axes=axes,
    )
    return X


def fft3c(X, axes=(-3, -2, -1), workers=-1):
    r"""3d FFT.

    Parameters
    ----------
    X : array_like
        Input array, can be complex.
    axes : int, default (-3, -2, -1)
        Axes over which to compute the FFT. If not given, the last three axes are used.

    Returns
    -------
    K : array_like
        K-space data.
    """
    K = fftshift(
        fftn(ifftshift(X, axes=axes), axes=axes, norm="ortho", workers=workers),
        axes=axes,
    )
    return K


def ifft3c(K, axes=(-3, -2, -1), workers=-1):
    r"""3d iFFT.

    Parameters
    ----------
    K : array_like
        Input K space data, can be complex.
    axes : int, default (-3, -2, -1)
        Axes over which to compute the iFFT. If not given, the last three axes are used.

    Returns
    -------
    X : array_like
        Original data.
    """
    X = fftshift(
        ifftn(ifftshift(K, axes=axes), axes=axes, norm="ortho", workers=workers),
        axes=axes,
    )
    return X


def fftnc(X, axes=None, workers=-1):
    r"""Nd FFT.

    Parameters
    ----------
    X : array_like
        Input array, can be complex.

    Returns
    -------
    K : array_like
        K-space data.
    """
    K = fftshift(fftn(ifftshift(X, axes=axes), axes=axes, norm="ortho", workers=workers), axes=axes)
    return K


def ifftnc(K, axes=None, workers=-1):
    r"""Nd iFFT.

    Parameters
    ----------
    K : array_like
        Input K space data, can be complex.

    Returns
    -------
    X : array_like
        Original data.
    """
    X = fftshift(ifftn(ifftshift(K, axes=axes), axes=axes, norm="ortho", workers=workers), axes=axes)
    return X


def zcrop(X, shape):
    """Pad or crop a ND array around its center.

    The center is defined as floor(N/2) if N is odd, or N/2 if N is even.

    Parameters
    ----------
    X : array_like
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
