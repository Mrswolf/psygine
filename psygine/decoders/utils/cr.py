# -*- coding: utf-8 -*-
# Authors: swolf <swolfforever@gmail.com>
# Date: 2022/11/28
# License: MIT License
"""Basic methods related with CR decomposition.
"""
import numpy as np

__all__ = [
    'rref', 'cr'
]

def rref(A):
    r"""Reduced row echelon form.

    Parameters
    ----------
    A : (m, n) array_like
        Any matrix.

    Returns
    -------
    A : (m, k) array_like
        The reduced row echelon form of A.
    jb : array_like
        The pivot column index.
    """
    A = np.copy(A)
    m, n = A.shape
    i, j = 0, 0
    jb = []
    tol = np.maximum(m, n)*np.finfo(np.float64).eps*np.linalg.norm(A, ord=np.inf)
    while i < m and j < n:
        # find index and value of the largest element in the remainder of column
        k = np.argmax(np.abs(A[i:m, j])) + i
        v = np.abs(A[k, j])
        if v <= tol:
            # the column is negligible, zero it out
            A[i:m, j] = 0
            j += 1
        else:
            # rember the pivot column index
            jb.append(j)
            # swap the i-th and k-th rows
            A[[i, k]] = A[[k, i]]
            # divide the pivot row(i) by the pivot element A[i, j]
            A[i, j:n]  = A[i, j:n] / A[i, j]
            # substract multiples of the pivot row from all the other rows
            for k in range(m):
                if k != i:
                    A[k, j:n] -= A[k, j]*A[i, j:n]
            j += 1
            i += 1
    return A, np.array(jb)

def cr(A):
    r"""Column-Row factorization.

    Parameters
    ----------
    A : (m, n) array_like
        Any matrix.

    Returns
    -------
    C : (m, k) array_like
        The column space of A, :math:`k \leq min(m, n)`.
    R : (k, n) array_like
        The row space of A.

    Notes
    -----
    The Column-Row factorization from Gilbert Strange, that expresses any matrix as a product of a matrix that describes its column space and a matrix describes its row space:

    .. math:: \mathbf{A} = \mathbf{C} \mathbf{R}

    where :math:`\mathbf{R}` is the reduced row echelon form and :math:`\mathbf{C}` is just independent columns of :math:`\mathbf{A}`. Since the reduced row echelon form is unique, CR factorization of any matrix is unique.

    The only defect of this implementation is that the reduced row echelon form may encounter roundoff errors, leading to a different value for the rank.

    CR factorization may not be a contender for any serious technical use, but it's quite interesting. See more information about CR factorization in [1]_ and [2]_.

    Examples
    --------
    >>> A = np.array([[1,2,1,2],[1,2,2,3],[1,2,3,4],[1,2,4,5]])
    >>> C, R = cr(A)
    >>> B = np.matmul(C, R)

    References
    ----------
    .. [1] https://www.norbertwiener.umd.edu/FFT/2020/Faraway%20Slides/Faraway%20Strang.pdf
    .. [2] https://blogs.mathworks.com/cleve/2020/10/23/gil-strang-and-the-cr-matrix-factorization/#4733a7bc-91f3-4c3a-8a51-6b7918e2d566
    """
    R, j = rref(A)
    r = len(j) # the rank of A
    R = R[:r, :] # the reduced form of R
    C = A[:, j]
    return C, R
