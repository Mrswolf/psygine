# -*- coding: utf-8 -*-
# Authors: swolf <swolfforever@gmail.com>
# Date: 2022/11/11
# License: MIT License
"""Basic methods related with fourier transform.
"""
from numpy.fft import fft, ifft, fft2, ifft2, fftn, ifftn, fftshift, ifftshift

__all__ = [
    'fft1c', 'ifft1c', 'fft2c', 'ifft2c', 'fftnc', 'ifftnc'
]

def fft1c(X, axis=-1):
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
    K = fftshift(fft(ifftshift(X, axes=axis), axis=axis, norm='ortho'), axes=axis)
    return K

def ifft1c(K, axis=-1):
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
    X = ifftshift(ifft(fftshift(K, axes=axis), axis=axis, norm='ortho'), axes=axis)
    return X

def fft2c(X, axes=(-2, -1)):
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
    K = fftshift(fft2(ifftshift(X, axes=axes), axes=axes, norm='ortho'), axes=axes)
    return K

def ifft2c(K, axes=(-2, -1)):
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
    X = ifftshift(ifft2(fftshift(K, axes=axes), axes=axes, norm='ortho'), axes=axes)
    return X

def fftnc(X):
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
    K = fftshift(fftn(ifftshift(X), norm='ortho'))
    return K

def ifftnc(K):
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
    X = ifftshift(ifftn(fftshift(K), norm='ortho'))
    return X

