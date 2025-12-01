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
from scipy.linalg import eigh
from joblib import Parallel, delayed
from psygine.decoders.utils import fftnc, ifftnc, zcrop, fastsvd
from psygine.decoders.mri.cc import calcSCCMtx

__all__ = ["espirit", "construct_hankel", "sum_of_diags"]


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
    Nc, Nx, Ny, Nz = calib_data.shape

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


def sum_of_diags(A, axis1=-2, axis2=-1, L2R=True):
    """Compute the sum of diagonals and subdiagonals of a matrix

    Parameters
    ----------
    A : array_like
        Input matrix of shape (..., M, ..., M, ...).
    axis1 : int, optional
        First axis to sum over. Default is -2.
    axis2 : int, optional
        Second axis to sum over. Default is -1.
    L2R: bool, optional
        If True, the sums of diagonals are computed from the left bottom to the
        right top, else the sums of diagonals are computed from the right top to the
        left bottom. Default is True.

    Returns
    -------
    B : array_like
        The sum of the diagonals and subdiagonals of A, with shape (..., 2 * M - 1, ...)
        Values are put along the axis2, from the right top to the left bottom.
    """
    Ndim = A.ndim
    if Ndim < 2:
        raise ValueError("A must be at least 2-dimensional.")
    if A.shape[axis1] != A.shape[axis2]:
        raise ValueError("The two dimensions of A must be equal.")
    M = A.shape[axis2]
    new_shape = list(np.shape(A))
    new_shape[axis2] = 2 * M - 1
    del new_shape[axis1]

    B = np.zeros_like(A, shape=tuple(new_shape))
    flag = 1 if L2R else -1
    for i in range(-M + 1, M):
        new_indices = [
            slice(None) if j != np.mod(axis2, Ndim) else i + M - 1 for j in range(Ndim)
        ]
        del new_indices[axis1]
        B[tuple(new_indices)] = np.trace(A, offset=flag * i, axis1=axis1, axis2=axis2)

    return B


def espirit(
    k_data,
    calib_size=[24, 24, 24],
    kernel_size=[6, 6, 6],
    map_dims=None,
    ns_threshold=1e-3,
    n_maps=1,
    crop_threshold=0.8,
    softcrop=False,
    rotphase=False,
    rotmethod='pca',
    normalize=False,
    n_jobs=-1
):
    """ESPIRiT, improved by using the K matrix.

    Parameters
    ----------
    k_data : array_like
        Multi-channel k-space data of shape (Nc, Nx, Ny, Nz), where Nc
        is the number of channels and (Nx, Ny, Nz) are the spatial dimensions.
    calib_size : list of int, optional
        Calibration region size. Default is [24, 24, 24].
    kernel_size : list of int, optional
        Size of the kernel. Default is [6, 6, 6].
    map_dims : list of int, optional
        Dimensions of the output maps. If None, it will be set to (Nx, Ny, Nz).
    ns_threshold : float, optional
        Threshold for the singular values. Singular values below this threshold
        times the largest singular value are set to zero. Default is 1e-3.
    n_maps : int, optional
        Number of maps to compute. Default is 1.
    crop_threshold : float, optional
        Threshold for cropping the maps. Values below this threshold are set to zero.
        Default is 0.8.
    softcrop : bool, optional
        If True, a soft cropping is applied to the maps. Default is False.
    rotphase : bool, optional
        If True, the phase ambiguity is removed by rotmethod.
        Default is False.
    rotmethod: str, optional
        Avaliable options, 'pca', 'first', 'vcc'. Default is 'pca'.
    normalize : bool, optional
        If True, the maps are normalized with L1-norm. Default is False.

    Returns
    -------
    maps : array_like
        ESPIRiT maps of shape (n_maps, Nc, Nx, Ny, Nz), where n_maps is the number of maps,
        Nc is the number of channels, and (Nx, Ny, Nz) are the spatial dimensions.
    """
    Nc, Nx, Ny, Nz = k_data.shape
    calib_data = zcrop(k_data, (Nc, *calib_size))
    if map_dims is None:
        map_dims = (Nx, Ny, Nz)

    rot_vec = None
    if rotphase:
        if rotmethod == 'pca':
            rot_vec = calcSCCMtx(calib_data, axis=0)
            rot_vec = np.conj(rot_vec[:, 0]).T
        elif rotmethod == 'first':
            rot_vec = np.zeros((1, Nc), dtype=k_data.dtype)
            rot_vec[0] = 1
        elif rotmethod == 'vcc':
            calib_data = np.concatenate(
                (calib_data, 
                 np.flip(np.flip(np.flip(np.conj(calib_data), axis=1), axis=2), axis=3)),
                 axis=0)
            Nc *= 2
        else:
            raise NotImplementedError(f"rotmethod {rotmethod} is not implemented.")

    A = construct_hankel(calib_data, kernel_size)
    U, S, Vh = fastsvd(A, method="matlab")

    n_kernels = np.sum(S >= ns_threshold * S[0])
    # subspace
    Vh = Vh[:n_kernels, :]
    P = Vh.T @ np.conj(Vh)
    P = np.reshape(P, (Nc, *kernel_size, Nc, *kernel_size))
    P = np.transpose(P, (0, 4, 1, 5, 2, 6, 3, 7))  # (Nc, Nc, Kx, Kx, Ky, Ky, Kz, Kz)
    K = sum_of_diags(
        sum_of_diags(sum_of_diags(P, L2R=False), axis1=-3, axis2=-2, L2R=False),
        axis1=-4,
        axis2=-3,
        L2R=False,
    )
    K = zcrop(K, (Nc, Nc, *map_dims))
    kerimgs = ifftnc(K, axes=(-3, -2, -1)) # see my blog post why we use ifftnc here
    # M^{-1/2}N^{1/2}, a constant factor
    kerimgs *= np.sqrt(np.prod(map_dims)) / np.sqrt(np.prod(kernel_size))

    def _espirit_maps(
        i, j, k, n_components=1, rotphase=False, rot_vec=None, crop_threshold=0, softcrop=False
    ):
        GqGqH = kerimgs[..., i, j, k]

        s, U = eigh(GqGqH)
        U, s = U[:, ::-1], s[::-1]
        U = U[:, :n_components]

        if rotphase:
            if rot_vec is None:
                # remove the phase ambiguity by vcc-espirit
                Nc = U.shape[0]
                theta = np.imag(np.log(np.sum(U[:Nc//2, :] * U[Nc//2:, :], axis=0))) / 2 # theta + kpi
                scale = np.cos(theta) - 1j * np.sin(theta)
                U = U[:Nc//2, :] * scale
            else:
                # remove the phase ambiguity 
                # by using either a virtual coil or the first coil
                scale = rot_vec @ U
                scale /= np.abs(scale) + np.finfo(scale.dtype).eps
                U *= np.conj(scale)

        weight = np.abs(U) >= crop_threshold
        if softcrop:
            weight = (np.sqrt(np.abs(U)) - crop_threshold) / (1 - crop_threshold)
            weight[weight < -1] = -1
            weight[weight > 1] = 1
            weight = weight / (np.square(weight) + 1) + 0.5
        U *= weight

        return U

    # Take the point-wise eigenvalue decomposition and keep eigenvalues greater than c
    maps = np.stack(
        Parallel(n_jobs=n_jobs)(
            delayed(_espirit_maps)(
                i,
                j,
                k,
                n_components=n_maps,
                rotphase=rotphase,
                rot_vec=rot_vec,
                crop_threshold=crop_threshold,
                softcrop=softcrop,
            )
            for i in range(map_dims[0])
            for j in range(map_dims[1])
            for k in range(map_dims[2])
        ),
        axis=0,
    )

    if rotphase and rotmethod == 'vcc':
        # sign alignment
        Nc //= 2
        calib_data = zcrop(calib_data[:Nc], (Nc, Nx, Ny, Nz))
        calib_data = ifftnc(calib_data, axes=(1, 2, 3))
        maps = np.reshape(maps, (*map_dims, Nc, n_maps))
        maps = np.transpose(maps, (4, 3, 0, 1, 2))
        # the original bart script computed the inner product of maps and calib_data, summed over the coil axis and took the sign of the real part as an estimation
        ref = np.real(np.sum(maps * np.conj(calib_data), axis=1, keepdims=True))
        ref = np.sign(ref)
        maps *= ref
        
    else:
        maps = np.reshape(maps, (*map_dims, Nc, n_maps))
        maps = np.transpose(maps, (4, 3, 0, 1, 2))
    
    if normalize:
        # Normalize the maps with L1-norm, maybe wrong?
        maps = maps / np.linalg.norm(maps, ord=1, axis=1, keepdims=True)
    return maps
