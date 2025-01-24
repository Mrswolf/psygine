# -*- coding: utf-8 -*-
# Authors: swolf <swolfforever@gmail.com>
# Date: 2023/01/01
# License: MIT License
"""Basic methods related with covariance.
"""
from functools import partial
import numpy as np
from scipy.linalg import eigh
from sklearn.covariance import oas, ledoit_wolf, fast_mcd, empirical_covariance
from joblib import Parallel, delayed

__all__ = [
    "is_positive_definite",
    "nearest_positive_definite",
    "covariances",
    "positive_definite_operator",
    "sqrtm",
    "invsqrtm",
    "logm",
    "expm",
    "powm",
]


def is_positive_definite(A):
    r"""Determine if the input matrix is positive-definite.

    Parameters
    ----------
    A : (n, n) array_like
        Any square matrix.

    Returns
    -------
    bool
        Return True if P is positive-definite.

    Notes
    -----
    Use cholesky decomposition to determine if the matrix is positive-definite.

    Examples
    --------
    >>> P = np.random.randn(50, 6)
    >>> P = np.matmul(P, P.T)
    >>> is_positive_definite(P)
    >>> P[0] = 0
    >>> is_positive_definite(P)
    """
    try:
        _ = np.linalg.cholesky(A)
        return True
    except np.linalg.LinAlgError:
        return False


def nearest_positive_definite(A):
    r"""Find the nearest postive-definite matrix to the input matrix.

    Parameters
    ----------
    A : (n, n) array_like
        Any square matrix.

    Returns
    -------
    A3
        The nearest positive-definite matrix to A.

    Notes
    -----
    A Python version of John D'Errico's `nearestSPD` MATLAB code [1]_, which
    origins at [2]_.

    Examples
    --------
    >>> A = np.random.randn(50, 6)
    >>> P = np.matmul(P, P.T)
    >>> P1 = P
    >>> P1 = nearest_positive_definite(P)

    References
    ----------
    .. [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd
    .. [2] Higham, Nicholas J. "Computing a nearest symmetric positive semidefinite matrix." Linear algebra and its applications 103 (1988): 103-118.
    """
    B = (A + A.T) / 2
    _, s, V = np.linalg.svd(B)
    H = np.dot(V.T, np.dot(np.diag(s), V))
    A2 = (B + H) / 2
    A3 = (A2 + A2.T) / 2
    if is_positive_definite(A3):
        return A3
    spacing = np.spacing(np.linalg.norm(A))
    # The above is different from [1]. It appears that MATLAB's `chol` Cholesky
    # decomposition will accept matrixes with exactly 0-eigenvalue, whereas
    # Numpy's will not. So where [1] uses `eps(mineig)` (where `eps` is Matlab
    # for `numpy.spacing`), we use the above definition. CAVEAT: our `spacing`
    # will be much larger than [1]'s `eps(mineig)`, since `mineig` is usually on
    # the order of 1e-16, and `eps(1e-16)` is on the order of 1e-34, whereas
    # `spacing` will, for Gaussian random matrixes of small dimension, be on
    # other order of 1e-16. In practice, both ways converge.
    I = np.eye(A.shape[0])
    k = 1
    while not is_positive_definite(A3):
        mineig = np.min(np.real(np.linalg.eigvals(A3)))
        A3 += I * (-mineig * k**2 + spacing)
        k += 1
    return A3


def _lwf(X):
    r"""Wrapper for sklearn ledoit wolf covariance estimator.

    Parameters
    ----------
    X : (n, m) array_like
        Data from which to compute the covariance estimate, where m is the number of samples and n is the number of features.

    Returns
    -------
    C : (n, n) array_like
        Estimated covariance.
    """
    C, _ = ledoit_wolf(X.T)
    return C


def _oas(X):
    """Wrapper for sklearn oas covariance estimator.

    Parameters
    ----------
    X : (n, m) array_like
        Data from which to compute the covariance estimate, where m is the number of samples and n is the number of features.

    Returns
    -------
    C : (n, n) array_like
        Estimated covariance.
    """
    C, _ = oas(X.T)
    return C


def _mcd(X):
    """Wrapper for sklearn mcd covariance estimator.

    Parameters
    ----------
    X : (n, m) array_like
        Data from which to compute the covariance estimate, where m is the number of samples and n is the number of features.

    Returns
    -------
    C : (n, n) array_like
        Estimated covariance.
    """
    _, C, _, _ = fast_mcd(X.T)
    return C


def _cov(X):
    """Wrapper for sklearn empirical covariance estimator.

    Parameters
    ----------
    X : (n, m) array_like
        Data from which to compute the covariance estimate, where m is the number of samples and n is the number of features.

    Returns
    -------
    C : (n, n) array_like
        Estimated covariance.
    """
    C = empirical_covariance(X.T)
    return C


_covariance_estimators = {
    "cov": _cov,
    "lwf": _lwf,
    "oas": _oas,
    "mcd": _mcd,
}


def _check_cov_est(est):
    r"""Check if a given covariance estimator is valid.

    Parameters
    ----------
    est : callable object or str
        Could be the name of estimator or a callable estimator itself.

    Returns
    -------
    est: callable object
        A callable estimator.
    """
    if callable(est):
        pass
    elif est in _covariance_estimators:
        est = _covariance_estimators[est]
    else:
        raise ValueError(
            """%s is not an valid estimator ! Valid estimators are : %s or a
             callable function"""
            % (est, (" , ").join(_covariance_estimators.keys()))
        )
    return est


def covariances(X, estimator="cov", n_jobs=None):
    r"""Covariance matrices of the inputs.

    Parameters
    ----------
    X : (..., n, m) array_like
        Data from which to compute the covariance estimate, where m is the number of samples and n is the number of features.
        The last two dimensions are used to compute covariance.

    estimator : str or callable, default 'cov'
        Covariance estimator to use, the default is `cov`, which uses empirical covariance estimator.
        For regularization purpose, consider `lwf` or `oas`.

        **supported estimators**

            `cov`: empirial covariance estimator

            `lwf`: ledoit wolf covariance estimator

            `oas`: oracle approximating shrinkage covariance estimator

            `mcd`: minimum covariance determinant covariance estimator
    n_jobs : int, optional
        The number of CPUs to use to do the computation, -1 for all cores.
        Default is None, which uses 1 core.

    Returns
    -------
    covmats : (..., n, n) array_like
        Estimated covariance matrices.
    """
    X = np.asarray(X)
    X = np.atleast_2d(X)
    shape = X.shape
    X = np.reshape(X, (-1, shape[-2], shape[-1]))

    parallel = Parallel(n_jobs=n_jobs)
    est = _check_cov_est(estimator)
    covmats = parallel(delayed(est)(x) for x in X)
    covmats = np.reshape(covmats, (*shape[:-2], shape[-2], shape[-2]))
    return covmats


def positive_definite_operator(P, operator, n_jobs=None):
    r"""Apply matrix operator to postive-definite matrices.

    Parameters
    ----------
    P : (..., n, n) array_like
        Positive-definite matrices, usually covariances.
    operator : callable
        Any callable object or function on eigen values of a positive-definite matrix.
    n_jobs: int, optional
        The number of cores to do the computation, default None.

    Returns
    -------
    P_hat : (..., n, n) array_like
        Operated matrices.

    Raises
    ------
    ValueError
        If P is not positive-definite.

    Notes
    -----
    For any positive-definite matrix :math:`\mathbf{Ci}`, the operator does the following math:

    .. math::
        \mathbf{V} operator\left( \mathbf{\Lambda} \\right) \mathbf{V}^T

    where :math:`\mathbf{\Lambda}` is the eigenvalues and :math:`\mathbf{V}` is the eigenvectors of :math:`\mathbf{Ci}`.
    """

    def _single_matrix_operator(Ci, operator):
        if not is_positive_definite(Ci):
            raise ValueError("The input matrix should be positive-definite.")
        eigvals, eigvects = eigh(Ci)
        # rtol = np.spacing(np.linalg.norm(eigvals, ord=np.inf)) * np.max(Ci.shape)
        # r1 = eigvals > rtol
        eigvals = np.diag(operator(eigvals))
        #eigvals[np.logical_not(r1)] = 0
        Co = eigvects @ eigvals @ eigvects.T
        return Co

    shape = P.shape
    P = P.reshape((-1, *shape[-2:]))
    P_hat = Parallel(n_jobs=n_jobs)(
        delayed(_single_matrix_operator)(Ci, operator) for Ci in P
    )
    P_hat = np.reshape(P_hat, (*shape,))
    return P_hat


def sqrtm(P, n_jobs=None):
    r"""Return the matrix square root of positive-definite matrices.

    Parameters
    ----------
    P : (..., n, n) array_like
        Positive-definite matrices.
    n_jobs: int, optional
        The number of cores to do the computation, default None.

    Returns
    -------
    (..., n, n) array_like
        Square root matrix of P.

    Notes
    -----
    For any positive-definite matrix :math:`\mathbf{Ci}`, the operator does the following math:

    .. math::
        \mathbf{V} \left( \mathbf{\Lambda} \\right)^{1/2} \mathbf{V}^T

    where :math:`\mathbf{\Lambda}` is the eigenvalues and :math:`\mathbf{V}` is the eigenvectors of :math:`\mathbf{Ci}`.
    """
    return positive_definite_operator(P, np.sqrt, n_jobs=n_jobs)


def invsqrtm(P, n_jobs=None):
    r"""Return the inverse matrix square root of positive-definite matrices.

    Parameters
    ----------
    P : (..., n, n) array_like
        Positive-definite matrices.

    Returns
    -------
    (..., n, n) array_like
        Inverse matrix square root of P.

    Notes
    -----
    For any positive-definite matrix :math:`\mathbf{Ci}`, the operator does the following math:

    .. math::
        \mathbf{V} \left( \mathbf{\Lambda} \\right)^{-1/2} \mathbf{V}^T

    where :math:`\mathbf{\Lambda}` is the eigenvalues and :math:`\mathbf{V}` is the eigenvectors of :math:`\mathbf{Ci}`.
    """
    isqrt = lambda x: 1.0 / np.sqrt(x)
    return positive_definite_operator(P, isqrt, n_jobs=n_jobs)


def logm(P, n_jobs=None):
    r"""Return the matrix logrithm of positive-definite matrices.

    Parameters
    ----------
    P : (..., n, n) array_like
        Positive-definite matrices.

    Returns
    -------
    (..., n, n) array_like
        Logrithm matrix of P.

    Notes
    -----
    For any positive-definite matrix :math:`\mathbf{Ci}`, the operator does the following math:

    .. math::
        \mathbf{V} \log{(\mathbf{\Lambda})} \mathbf{V}^T

    where :math:`\mathbf{\Lambda}` is the eigenvalues and :math:`\mathbf{V}` is the eigenvectors of :math:`\mathbf{Ci}`.
    """
    return positive_definite_operator(P, np.log, n_jobs=n_jobs)


def expm(P, n_jobs=None):
    r"""Return the matrix exponential of positive-definite matrices.

    Parameters
    ----------
    P : (..., n, n) array_like
        Positive-definite matrices.
    n_jobs: int, optional
        The number of cores to do the computation, default None.

    Returns
    -------
    (..., n, n) array_like
        Exponential matrix of P.

    Notes
    -----
    For any positive-definite matrix :math:`\mathbf{Ci}`, the operator does the following math:

    .. math::
        \mathbf{C} = \mathbf{V} \exp{(\mathbf{\Lambda})} \mathbf{V}^T

    where :math:`\mathbf{\Lambda}` is the eigenvalues and :math:`\mathbf{V}` is the eigenvectors of :math:`\mathbf{Ci}`.
    """
    return positive_definite_operator(P, np.exp, n_jobs=n_jobs)


def powm(P, alpha, n_jobs=None):
    r"""Return the matrix power of positive-definite matrices.

    Parameters
    ----------
    P : (..., n, n) array_like
        Positive-definite matrices.
    alpha : float
        Exponent.
    n_jobs: int, optional
        The number of cores to do the computation, default None.

    Returns
    -------
    (..., n, n) array_like
        Power matrix of P.

    Notes
    -----
    For any positive-definite matrix :math:`\mathbf{Ci}`, the operator does the following math:

    .. math::
        \mathbf{V} \left( \mathbf{\Lambda} \\right)^{\\alpha} \mathbf{V}^T

    where :math:`\mathbf{\Lambda}` is the eigenvalues and :math:`\mathbf{V}` is the eigenvectors of :math:`\mathbf{Ci}`.
    """
    power = partial(lambda x, alpha=None: x**alpha, alpha=alpha)
    return positive_definite_operator(P, power, n_jobs=n_jobs)
