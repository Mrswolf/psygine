# -*- coding: utf-8 -*-
# Authors: swolf <swolfforever@gmail.com>
# Date: 2025/01/26
# License: MIT License
"""Espirit.

Modified from
https://github.com/mikgroup/espirit-python/blob/master/espirit.py
""" 
import time
import numpy as np
from psygine.decoders.utils.fourier import fftnc, ifftnc
from scipy.linalg import svd, eigh
from joblib import Parallel, delayed

__all__ =['espirit', 'espirit_proj3d']

def get_calib(X, r):
    Ns = np.shape(X)

    slices = []
    # floor(N/2) is the center index
    for Ncalib, N in zip(r, Ns):
        if Ncalib > N:
            Ncalib = N
        startIdx = N//2-Ncalib//2
        endIdx = startIdx + Ncalib
        slices.append(slice(startIdx, endIdx))
    slices = tuple(slices)
    return X[slices]

def set_calib(X, C):
    Ns = np.shape(X)
    r = np.shape(C)

    slices = []
    for Ncalib, N in zip(r, Ns):
        if Ncalib > N:
            raise ValueError(f"{Ncalib} is larger than {N}")
        startIdx = N//2-Ncalib//2
        endIdx = startIdx + Ncalib
        slices.append(slice(startIdx, endIdx))
    slices = tuple(slices)
    X[slices] = C


def construct_hankel(calib_data, kernel_size):
    """Construct Block-Hankel matrix from calibration data.

    Parameters
    ----------
    calib_data : array_like
        Calibration data of shape (Nc, Nx, Ny, Nz), where Nc is the number of channels.
    kernel_size : array_like
        Size of the kernel. Should be of shape (Kx, Ky, Kz).

    Returns
    -------
    A : array_like
        Block-Hankel matrix of shape (N_kernels, np.prod(kernel_size) * Nc), where N_kernels is the number of kernels.
        Local pathches are flattened along the last axis.
    """
    [Nc, Nx, Ny, Nz] = calib_data.shape

    kernel_dims = np.array([Nx, Ny, Nz]) - np.array(kernel_size) + 1
    Nk = np.prod(kernel_dims)
    A = np.zeros((Nk, np.prod(kernel_size) * Nc), dtype=calib_data.dtype)

    # let numpy handle the indexing
    for i in range(Nk):
        # slide along the last axis
        local_idxs = np.unravel_index(i, kernel_dims, order="C")
        local_slices = [slice(0, Nc)]
        local_slices.extend(
            [slice(idx, idx + kernel_size[i]) for i, idx in enumerate(local_idxs)]
        )
        local_slices = tuple(local_slices)
        # flatten the local patch along the last axis
        A[i, :] = calib_data[local_slices].flatten()

    return A


def espirit(calib_data, kernel_size, map_dims=None, t=0.01, n_components=1):
    """ESPIRiT.

    """
    stime = time.time()
    [Nc, Nx, Ny, Nz] = np.shape(calib_data)
    if map_dims is None:
        map_dims = (Nx, Ny, Nz)

    A = construct_hankel(calib_data, kernel_size)
    [U, S, Vh] = svd(A, full_matrices=True)

    n_kernels = np.sum(S >= t*S[0])
    # subspace
    Vh = Vh[:n_kernels, :]

    # Reshape into k-space kernel
    kernels = np.reshape(Vh, (n_kernels, Nc, *kernel_size))
    # The kernels are in the form of (Nk, Nc, Kx, Ky, Kz)
    # centering the kernels and iFFT, which is the same as FFT of the conjugate
    # the kerimgs are in the form of (Nk, Nc, map_dims[0], map_dims[1], map_dims[2])
    kerimgs = np.zeros((n_kernels, Nc, *map_dims), dtype=calib_data.dtype)
    set_calib(kerimgs, kernels)
    kerimgs = fftnc(kerimgs, axes=(-3, -2, -1))
    # M^{-1/2}N^{1/2}
    kerimgs *= np.sqrt(np.prod(map_dims)) / np.sqrt(np.prod(kernel_size))

    def _espirit_maps(i, j, k, n_components=1):
        Gq = kerimgs[..., i, j, k]
        [s, U] = eigh(Gq.conj().T @ Gq)
        s = s[-n_components:]
        U = U[:, -n_components:]
        return U

    # Take the point-wise eigenvalue decomposition and keep eigenvalues greater than c
    maps = np.zeros((n_components, Nc, *map_dims), dtype=kerimgs.dtype)
    maps = np.stack(
        Parallel(n_jobs=1)(
            delayed(_espirit_maps)(i, j, k, n_components=n_components)
            for i in range(map_dims[0])
            for j in range(map_dims[1])
            for k in range(map_dims[2])
        ),
        axis=0,
    )
    maps = np.reshape(maps, (*map_dims, Nc, n_components))
    maps = np.transpose(maps, (4, 3, 0, 1, 2))
    return maps
