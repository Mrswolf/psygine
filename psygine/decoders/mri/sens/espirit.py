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
from scipy.linalg import svd
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

def construct_hankel(C, k):
    Ns = np.shape(C)[:-1]
    Nc = np.shape(C)[-1]
    
    N_kernel_eles = np.prod(k)
    N_kernel_dims = np.array(Ns) - np.array(k) + 1
    N_kernels = np.prod(N_kernel_dims)
    A = np.zeros((N_kernels, N_kernel_eles * Nc), dtype=C.dtype)

    for i in range(N_kernels):
        local_idxs = np.unravel_index(i, N_kernel_dims, order='C')
        local_slices = [slice(idx, idx+k[i]) for i, idx in enumerate(local_idxs)]
        local_slices.append(slice(0, Nc))
        local_slices = tuple(local_slices)

        A[i, :] = C[local_slices].flatten()

    return A

def espirit(X, k, r, t, c):
    """ESPIRiT.

    """
    stime = time.time()
    Ns = np.shape(X)[:-1]
    Nd = len(Ns)
    Nc = np.shape(X)[-1]

    if isinstance(r, int):
        r = [Ns[i] if Ns[i] < r else r for i in range(Nd)]
    else:
        for i in range(Nd):
            r[i] = Ns[i] if Ns[i] < r[i] else r[i]
    r.append(Nc)

    if isinstance(k, int):
        k = [r[i] if r[i] < k else k for i in range(Nd)]
    else:
        for i in range(Nd):
            k[i] = r[i] if r[i] < k[i] else k[i]

    C = get_calib(X, r)
    A = construct_hankel(C, k)
    [U, S, Vh] = svd(A, full_matrices=True)
    V = Vh.conj().T

    n_kernels = np.sum(S >= t*S[0])
    # subspace
    V = V[:, :n_kernels]

    # Reshape into k-space kernel
    kernels = np.zeros(np.append(k, (Nc, n_kernels)), dtype=V.dtype)
    for ik in range(n_kernels):
        kernels[..., ik] = np.reshape(V[:, ik], np.append(k, Nc))

    # flips it and takes the conjugate
    # But why?
    kerimgs = np.zeros(np.append(Ns, (Nc, n_kernels)), dtype=V.dtype)
    set_calib(kerimgs, kernels)

    N = np.prod(Ns)
    for i in range(n_kernels):
        for j in range(Nc):
            ker = kerimgs[..., j, i].conj()
            ker = np.flip(ker)
            kerimgs[..., j, i] = fftnc(ker) * np.sqrt(N) / np.sqrt(np.prod(k))

    def _espirit_maps(i, kerimgs):
        idxs = np.unravel_index(i, Ns)
        Gq = kerimgs[idxs]
        u, s, vh = svd(Gq, full_matrices=True)
        return u * (np.square(s) > c)

    # Take the point-wise eigenvalue decomposition and keep eigenvalues greater than c
    # maps = np.zeros((N, Nc, Nc), dtype=kerimgs.dtype)
    # for i in range(N):
    #     idxs = np.unravel_index(i, Ns)
    #     Gq = kerimgs[idxs]
    #     u, s, vh = svd(Gq, full_matrices=True)
    #     maps[i, ...] = u * (np.square(s) > c)
    maps = np.stack(Parallel(n_jobs=-1)(delayed(_espirit_maps)(i, kerimgs) for i in range(N)), axis=0)
    maps = np.reshape(maps, np.append(Ns, (Nc, Nc)))
    return maps

def espirit_proj3d(x, maps):
    """
    Construct the projection of multi-channel image x onto the range of the ESPIRiT operator. Returns the inner
    product, complete projection and the null projection.

    Arguments:
      x: Multi channel image data. Expected dimensions are (sx, sy, sz, nc), where (sx, sy, sz) are volumetric 
         dimensions and (nc) is the channel dimension.
      maps: ESPIRiT operator as returned by function: espirit

    Returns:
      ip: This is the inner product result, or the image information in the ESPIRiT subspace.
      proj: This is the resulting projection. If the ESPIRiT operator is E, then proj = E E^H x, where H is 
            the hermitian.
      null: This is the null projection, which is equal to x - proj.
    """
    ip = np.zeros(x.shape).astype(np.complex64)
    proj = np.zeros(x.shape).astype(np.complex64)
    for qdx in range(0, maps.shape[4]):
        for pdx in range(0, maps.shape[3]):
            ip[:, :, :, qdx] = ip[:, :, :, qdx] + x[:, :, :, pdx] * maps[:, :, :, pdx, qdx].conj()

    for qdx in range(0, maps.shape[4]):
        for pdx in range(0, maps.shape[3]):
          proj[:, :, :, pdx] = proj[:, :, :, pdx] + ip[:, :, :, qdx] * maps[:, :, :, pdx, qdx]

    return (ip, proj, x - proj)