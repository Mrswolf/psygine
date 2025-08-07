from typing import Optional
from joblib import Parallel, delayed
import numpy as np
from numpy import ndarray
from psygine.decoders.utils import fastsvd
from psygine.decoders.utils import ifft1c, fft1c

__all__ = ['calcSCCMtx', 'scc']

def scc(kdata: ndarray, A: ndarray, n_compressed_chs: int, axis: Optional[int] = 0):
    """Compress coils using the given matrix A.

    Parameters
    ----------
    kdata : ndarray
        Input kspace data.
    A : ndarray
        Compression matrix with shape (N_channels, N_compressed_channels).
    axis : int, default -1
        Dimension along which to compress the coils, default 0.

    Returns
    -------
    compressed_data : ndarray
        Compressed data.
    """
    kdata = np.movexis(kdata, axis, 0)
    shape = kdata.shape
    kdata = A[:,:n_compressed_chs].conj().T @ np.reshape(kdata, (shape[0], -1), order='C')
    kdata = np.reshape(kdata, (n_compressed_chs, *shape[1:]), order='C')
    kdata = np.moveaxis(kdata, 0, axis) 
    return kdata


def calcSCCMtx(
    kdata: ndarray, axis: Optional[int] = 0
) -> tuple[ndarray, ndarray]:
    """Calculate simple coil compression matrix.
    
    Parameters
    ----------
    kdata : ndarray
        Input kspace data.
    axis : Optional[int]
        Dimension along which to compress the coils, default -1.

    Returns
    -------
    A : ndarray
        Compression matrix, shape (N_channels, N_compressed_channels).

    """
    kdata = np.moveaxis(kdata, axis, 0)
    N_channels = kdata.shape[0]
    [A, _, _] = fastsvd(kdata.reshape((N_channels, -1), order="C"))
    return A

