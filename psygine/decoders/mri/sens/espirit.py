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
from scipy.linalg import svd, eigh
from joblib import Parallel, delayed
from psygine.decoders.utils import fftnc, ifftnc, zcrop, fastsvd

__all__ = ["espirit", "espirit_mikgroup"]


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
    normalize=False,
):
    Nc, Nx, Ny, Nz = k_data.shape
    calib_data = zcrop(k_data, (Nc, *calib_size))
    if map_dims is None:
        map_dims = (Nx, Ny, Nz)

    # TODO: remove phase ambiguity by using a virtual coil
    rot_mat = None
    # if rotphase:
    #     rot_mat, _ = scc(calib_data, estimator="lwf")
    #     rot_mat = rot_mat[:, 0].conj().T

    A = construct_hankel(calib_data, kernel_size)
    U, S, Vh = fastsvd(A, method="matlab")

    n_kernels = np.sum(S >= ns_threshold * S[0])
    # subspace
    Vh = Vh[:n_kernels, :]

    # Reshape into k-space kernel
    kernels = np.reshape(Vh, (n_kernels, Nc, *kernel_size))
    # The kernels are in the form of (Nk, Nc, Kx, Ky, Kz)
    # padding the kernels to match the map dimensions
    # the kerimgs are in the form of (Nk, Nc, map_dims[0], map_dims[1], map_dims[2])
    kerimgs = zcrop(kernels, (n_kernels, Nc, *map_dims))

    kerimgs = ifftnc(
        kerimgs, axes=(-3, -2, -1)
    )  # see my blog post why we use ifftnc here
    # kerimgs = fftnc(kerimgs[:, :, ::-1, ::-1, ::-1], axes=(-3, -2, -1))

    # M^{-1/2}N^{1/2}
    kerimgs *= np.sqrt(np.prod(map_dims)) / np.sqrt(np.prod(kernel_size))

    def _espirit_maps(
        i, j, k, n_components=1, rot_mat=None, crop_threshold=0, softcrop=False
    ):
        Gq = kerimgs[..., i, j, k]

        U, s, vh = fastsvd(Gq.T, k=n_components, method="matlab")

        if rot_mat is None:
            # remove the phase ambiguity by using the first coil
            ref = np.copy(U[0])
            ref /= np.abs(ref)
            U *= np.conj(ref)

        # TODO: remove phase ambiguity by using a virtual coil
        # # extract the phase from the first virtual coil
        # scale = rot_mat @ U
        # scale /= np.abs(scale)
        # U *= np.conj(scale)

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
        Parallel(n_jobs=-1)(
            delayed(_espirit_maps)(
                i,
                j,
                k,
                n_components=n_maps,
                rot_mat=rot_mat,
                crop_threshold=crop_threshold,
                softcrop=softcrop,
            )
            for i in range(map_dims[0])
            for j in range(map_dims[1])
            for k in range(map_dims[2])
        ),
        axis=0,
    )
    maps = np.reshape(maps, (*map_dims, Nc, n_maps))
    maps = np.transpose(maps, (4, 3, 0, 1, 2))
    if normalize:
        # Normalize the maps with L1-norm
        maps = maps / np.linalg.norm(maps, ord=1, axis=1, keepdims=True)
    return maps


# for comparison
def espirit_mikgroup(X, k, r, t, c):
    """
    Derives the ESPIRiT operator.

    Arguments:
      X: Multi channel k-space data. Expected dimensions are (sx, sy, sz, nc), where (sx, sy, sz) are volumetric
         dimensions and (nc) is the channel dimension.
      k: Parameter that determines the k-space kernel size. If X has dimensions (1, 256, 256, 8), then the kernel
         will have dimensions (1, k, k, 8)
      r: Parameter that determines the calibration region size. If X has dimensions (1, 256, 256, 8), then the
         calibration region will have dimensions (1, r, r, 8)
      t: Parameter that determines the rank of the auto-calibration matrix (A). Singular values below t times the
         largest singular value are set to zero.
      c: Crop threshold that determines eigenvalues "=1".
    Returns:
      maps: This is the ESPIRiT operator. It will have dimensions (sx, sy, sz, nc, nc) with (sx, sy, sz, :, idx)
            being the idx'th set of ESPIRiT maps.
    """

    sx = np.shape(X)[0]
    sy = np.shape(X)[1]
    sz = np.shape(X)[2]
    nc = np.shape(X)[3]

    sxt = (sx // 2 - r // 2, sx // 2 + r // 2) if (sx > 1) else (0, 1)
    syt = (sy // 2 - r // 2, sy // 2 + r // 2) if (sy > 1) else (0, 1)
    szt = (sz // 2 - r // 2, sz // 2 + r // 2) if (sz > 1) else (0, 1)

    # Extract calibration region.
    C = X[sxt[0] : sxt[1], syt[0] : syt[1], szt[0] : szt[1], :].astype(np.complex64)

    # Construct Hankel matrix.
    p = (sx > 1) + (sy > 1) + (sz > 1)
    A = np.zeros([(r - k + 1) ** p, k**p * nc]).astype(np.complex64)

    idx = 0
    for xdx in range(max(1, C.shape[0] - k + 1)):
        for ydx in range(max(1, C.shape[1] - k + 1)):
            for zdx in range(max(1, C.shape[2] - k + 1)):
                # numpy handles when the indices are too big
                block = C[xdx : xdx + k, ydx : ydx + k, zdx : zdx + k, :].astype(
                    np.complex64
                )
                A[idx, :] = block.flatten()
                idx = idx + 1

    # Take the Singular Value Decomposition.
    U, S, VH = np.linalg.svd(A, full_matrices=True)
    V = VH.conj().T

    # Select kernels.
    n = np.sum(S >= t * S[0])
    V = V[:, 0:n]

    kxt = (sx // 2 - k // 2, sx // 2 + k // 2) if (sx > 1) else (0, 1)
    kyt = (sy // 2 - k // 2, sy // 2 + k // 2) if (sy > 1) else (0, 1)
    kzt = (sz // 2 - k // 2, sz // 2 + k // 2) if (sz > 1) else (0, 1)

    # Reshape into k-space kernel, flips it and takes the conjugate
    kernels = np.zeros(np.append(np.shape(X), n)).astype(np.complex64)
    kerdims = [
        (sx > 1) * k + (sx == 1) * 1,
        (sy > 1) * k + (sy == 1) * 1,
        (sz > 1) * k + (sz == 1) * 1,
        nc,
    ]
    for idx in range(n):
        kernels[kxt[0] : kxt[1], kyt[0] : kyt[1], kzt[0] : kzt[1], :, idx] = np.reshape(
            V[:, idx], kerdims
        )

    # Take the iucfft
    axes = (0, 1, 2)
    kerimgs = np.zeros(np.append(np.shape(X), n)).astype(np.complex64)
    for idx in range(n):
        for jdx in range(nc):
            # ker = kernels[::-1, ::-1, ::-1, jdx, idx].conj()
            # kerimgs[:, :, :, jdx, idx] = (
            #     fftnc(ker, axes) * np.sqrt(sx * sy * sz) / np.sqrt(k**p)
            # )
            kerimgs[:, :, :, jdx, idx] = (
                ifftnc(np.conj(kernels[..., jdx, idx]), axes)
                * np.sqrt(sx * sy * sz)
                / np.sqrt(k**p)
            )

    # Take the point-wise eigenvalue decomposition and keep eigenvalues greater than c
    maps = np.zeros(np.append(np.shape(X), nc)).astype(np.complex64)
    for idx in range(0, sx):
        for jdx in range(0, sy):
            for kdx in range(0, sz):

                Gq = kerimgs[idx, jdx, kdx, :, :]

                u, s, vh = np.linalg.svd(Gq, full_matrices=True)
                for ldx in range(0, nc):
                    if s[ldx] ** 2 > c:
                        maps[idx, jdx, kdx, :, ldx] = u[:, ldx]

    return maps
