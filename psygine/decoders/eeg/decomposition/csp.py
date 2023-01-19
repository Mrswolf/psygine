# -*- coding: utf-8 -*-
# Authors: swolf <swolfforever@gmail.com>
# Date: 2023/01/18
# License: MIT License
"""CSP-related methods.
"""
import numpy as np
from scipy.linalg import eigh
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin, clone
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from psygine.decoders.utils import covariances
from .base import FilterBank

__all__ = [
    'csp_kernel', 'csp_feature', 'CSP', 'FBCSP', 'MultiCSP', 'FBMultiCSP'
]

def csp_kernel(X, y, cov_estimator='cov'):
    r"""CSP kernel.

    Parameters
    ----------
    X : (n_trials, n_channels, n_samples) array_like
        EEG signal.
    y : (n_trials,) array_like
        Labels.
    cov_estimator : str, default 'cov'
        Covariance estimator to use.

    Returns
    -------
    D : (n_channels,) array_like
        Eigenvalues in descending order.
    W : (n_channels, n_channels) array_like
        Eigenvectors corresponding to eigenvalues.

    Notes
    -----
    Instead of choosing the first and last m components in [1]_, this implementation sorted W with |D-0.5| in descending order, so that the first 2m components can be chosed.

    References
    ----------
    .. [1] Ramoser H, Muller-Gerking J, Pfurtscheller G. Optimal spatial filtering of single trial EEG during imagined hand movement[J]. IEEE transactions on rehabilitation engineering, 2000, 8(4): 441-446.
    """
    X = np.reshape(X, (-1, *X.shape[-2:]))
    labels = np.unique(y)
    if len(labels) != 2:
        raise ValueError("csp kernel is only for 2-class classification problem.")

    C1 = covariances(
        X[y==labels[0]],
        estimator=cov_estimator,
        n_jobs=1)
    C2 = covariances(
        X[y==labels[1]],
        estimator=cov_estimator,
        n_jobs=1)

    # trace normalization
    C1 = C1 / np.trace(C1, axis1=-1, axis2=-2)[..., np.newaxis, np.newaxis]
    C2 = C2 / np.trace(C2, axis1=-1, axis2=-2)[..., np.newaxis, np.newaxis]

    C1 = np.mean(C1, axis=0)
    C2 = np.mean(C2, axis=0)
    Cc = C1 + C2

    D, W = eigh(C1, Cc)
    # the eigenvalues range from 0 to 1
    # resorting with the 0.5 threshold
    D = np.abs(D - 0.5)
    idx = np.argsort(D)[::-1]

    D, W = D[idx], W[:,idx]
    return D, W

def csp_feature(X, W, n_components=2):
    r"""CSP feature.

    Parameters
    ----------
    X : (n_trials, n_channels, n_samples) array_like
        EEG signal.
    W : (n_channels, n_channels) array_like
        Spatial filters.
    n_components : int, default 2
        The number of components.

    Returns
    -------
    (n_trials, n_templates) array_like
        CSP features.

    Notes
    -----
    CSP features based on [1]_.

    References
    ----------
    .. [1] Ramoser H, Muller-Gerking J, Pfurtscheller G. Optimal spatial filtering of single trial EEG during imagined hand movement[J]. IEEE transactions on rehabilitation engineering, 2000, 8(4): 441-446.
    """
    X = np.reshape(X, (-1, *X.shape[-2:]))
    X = X - np.mean(X, axis=-1, keepdims=True)
    eps = np.finfo(X.dtype).eps
    feat = np.matmul(W[:,:n_components].T, X)
    # variance normalization
    feat = np.mean(np.square(feat), axis=-1)
    feat = feat / (np.sum(feat, axis=-1, keepdims=True) + eps)
    # log transformation
    feat = np.log(np.clip(feat, eps, None))
    return feat

class CSP(BaseEstimator, TransformerMixin, ClassifierMixin):
    r"""Common spatial pattern.

    Parameters
    ----------
    n_components : int, default 2
        The number of components.
    cov_estimator : str, default 'cov'
        Covariance estimator to use.
    classifier : estimator, optional
        Classifier in sklearn.
        If None, skips classification step.

    Notes
    -----
    CSP [1]_ method for MI-BCI.

    References
    ----------
    .. [1] Ramoser H, Muller-Gerking J, Pfurtscheller G. Optimal spatial filtering of single trial EEG during imagined hand movement[J]. IEEE transactions on rehabilitation engineering, 2000, 8(4): 441-446.
    """
    def __init__(self,
            n_components=2,
            cov_estimator='cov',
            classifier=None):
        self.n_components = n_components
        self.cov_estimator = cov_estimator
        self.classifier = classifier
        self.classifier_ = None

    def fit(self, X, y):
        r"""Fit to data.

        Parameters
        ----------
        X : (n_trials, n_channels, n_samples) array_like
            EEG signal.
        y : (n_trials,) array_like
            Labels.
        """
        X = np.reshape(X, (-1, *X.shape[-2:]))
        _, W = csp_kernel(
            X, y, cov_estimator=self.cov_estimator)
        self.W_ = W

        if self.classifier is not None:
            self.classifier_ = clone(self.classifier)
            feat = csp_feature(
                X, W, n_components=self.n_components)
            self.classifier_.fit(feat, y)
        return self

    def transform(self, X):
        r"""Transform data.

        Parameters
        ----------
        X : (n_trials, n_channels, n_samples) array_like
            EEG signal.

        Returns
        -------
        (n_trials, n_components) array_like
            CSP features.
        """
        X = np.reshape(X, (-1, *X.shape[-2:]))
        features = csp_feature(X, self.W_, n_components=self.n_components)
        return features

    def predict(self, X):
        r"""Predict data.

        Parameters
        ----------
        X : (n_trials, n_channels, n_samples) array_like
            EEG signal.

        Returns
        -------
        labels : (n_trials,) array_like
            Class labels.
        """
        feat = self.transform(X)
        if self.classifier_ is not None:
            labels = self.classifier_.predict(feat)
            return labels
        else:
            raise NotImplementedError("No classifier was provided to support predict function.")

class FBCSP(FilterBank, ClassifierMixin):
    r"""Filterbank CSP.

    Parameters
    ----------
    filterbank : list
        List of filter coefficients in sos form.
    n_components : int, default 1
        The number of components.
    cov_estimator : str, default 'cov'
        Covariance estimator to use.
    classifier : estimator, optional
        Classifier in sklearn.
        If None, skips classification step.
    n_mi_components : int, default 1
        The number of mutual information selection components.
    n_jobs : int, optional
        The number of CPUs to use to do the computation.

    Notes
    -----
    This implementation uses mutual information to select features, as described in [1]_.

    References
    ----------
    .. [1] Ang K K, Chin Z Y, Zhang H, et al. Filter bank common spatial pattern (FBCSP) in brain-computer interface[C]//2008 IEEE International Joint Conference on Neural Networks (IEEE World Congress on Computational Intelligence). IEEE, 2008: 2390-2397.
    """
    def __init__(self,
        filterbank,
        n_components=2,
        cov_estimator='cov',
        classifier=None,
        n_mi_components=4,
        n_jobs=None):
        self.filterbank = filterbank
        self.n_components = n_components
        self.n_mi_components = n_mi_components
        self.cov_estimator = cov_estimator
        self.classifier = classifier
        self.classifier_ = None
        self.n_jobs = n_jobs

        super().__init__(
            CSP(
                n_components=n_components,
                cov_estimator=cov_estimator,
                classifier=None),
            filterbank,
            concat=True,
            n_jobs=n_jobs)

    def fit(self, X, y):
        r"""Fit to data.

        Parameters
        ----------
        X : (n_trials, n_channels, n_samples) array_like
            EEG signal.
        y : (n_trials,) array_like
            Labels.
        """
        super().fit(X, y, axis=-1)

        # feature selection step
        feat = super().transform(X)
        self.selector_ = SelectKBest(
            score_func=mutual_info_classif,
            k=self.n_mi_components)
        self.selector_.fit(feat, y)

        if self.classifier is not None:
            feat = self.selector_.transform(feat)
            self.classifier_ = clone(self.classifier)
            self.classifier_.fit(feat, y)
        return self

    def transform(self, X):
        r"""Transform data.

        Parameters
        ----------
        X : (n_trials, n_channels, n_samples) array_like
            EEG signal.

        Returns
        -------
        (n_trials, n_mi_components) array_like
            CSP features with mutual information selection.
        """
        features = super().transform(X, axis=-1)
        features = self.selector_.transform(features)
        return features

    def predict(self, X):
        r"""Predict data.

        Parameters
        ----------
        X : (n_trials, n_channels, n_samples) array_like
            EEG signal.

        Returns
        -------
        labels : (n_trials,) array_like
            Class labels.
        """
        feat = self.transform(X)
        if self.classifier_ is not None:
            labels = self.classifier_.predict(feat)
            return labels
        else:
            raise NotImplementedError("No classifier was provided to support predict function.")

def _rjd(X, eps=1e-9, n_iter_max=1000):
    """Approximate joint diagonalization based on jacobi angle.

    Parameters
    ----------
    X : (n_trials, n_channels, n_channels) array_like
        A set of covariance matrices to diagonalize.
    eps : float, defaut 1e-9
        Tolerance for stopping criterion.
    n_iter_max : int, default 1000
        The maximum number of iteration to reach convergence.

    Returns
    -------
    V : (n_channels, n_filters) array_like
        The diagonalizer, usually n_filters == n_channels.
    D : (n_trials, n_channels, n_channels) array_like
        The set of quasi diagonal matrices.

    Notes
    -----
    This is a direct implementation of the Cardoso AJD algorithm [1]_ used in
    JADE. The code is a translation of the matlab code provided in the author
    website.

    References
    ----------
    .. [1] Cardoso, Jean-Francois, and Antoine Souloumiac. Jacobi angles for simultaneous diagonalization. SIAM journal on matrix analysis and applications 17.1 (1996): 161-164.
    """
    # reshape input matrix
    A = np.concatenate(X, 0).T

    # init variables
    m, nm = A.shape
    V = np.eye(m)
    encore = True
    k = 0

    while encore:
        encore = False
        k += 1
        if k > n_iter_max:
            break
        for p in range(m - 1):
            for q in range(p + 1, m):

                Ip = np.arange(p, nm, m)
                Iq = np.arange(q, nm, m)

                # computation of Givens angle
                g = np.array([A[p, Ip] - A[q, Iq], A[p, Iq] + A[q, Ip]])
                gg = np.dot(g, g.T)
                ton = gg[0, 0] - gg[1, 1]
                toff = gg[0, 1] + gg[1, 0]
                theta = 0.5 * np.arctan2(toff, ton +
                                         np.sqrt(ton * ton + toff * toff))
                c = np.cos(theta)
                s = np.sin(theta)
                encore = encore | (np.abs(s) > eps)
                if (np.abs(s) > eps):
                    tmp = A[:, Ip].copy()
                    A[:, Ip] = c * A[:, Ip] + s * A[:, Iq]
                    A[:, Iq] = c * A[:, Iq] - s * tmp

                    tmp = A[p, :].copy()
                    A[p, :] = c * A[p, :] + s * A[q, :]
                    A[q, :] = c * A[q, :] - s * tmp

                    tmp = V[:, p].copy()
                    V[:, p] = c * V[:, p] + s * V[:, q]
                    V[:, q] = c * V[:, q] - s * tmp

    D = np.reshape(A, (m, int(nm / m), m)).transpose(1, 0, 2)
    return V, D

def _ajd_pham(X, eps=1e-9, n_iter_max=1000):
    """Approximate joint diagonalization based on pham's algorithm.

    Parameters
    ----------
    X : (n_trials, n_channels, n_channels) array_like
        A set of covariance matrices to diagonalize.
    eps : float, default 1e-6
        Tolerance for stoping criterion.
    n_iter_max : int, default 1000
        The maximum number of iteration to reach convergence.

    Returns
    -------
    V : (n_channels, n_filters) array_like
        The diagonalizer, usually n_filters == n_channels.
    D : (n_trials, n_channels, n_channels) array_like
        The set of quasi diagonal matrices.

    Notes
    -----
    This is a direct implementation of the PHAM's AJD algorithm [1]_.

    References
    ----------
    .. [1] Pham, Dinh Tuan. "Joint approximate diagonalization of positive definite Hermitian matrices." SIAM Journal on Matrix Analysis and Applications 22, no. 4 (2001): 1136-1152.
    """
    # Adapted from http://github.com/alexandrebarachant/pyRiemann
    n_epochs = X.shape[0]

    # Reshape input matrix
    A = np.concatenate(X, axis=0).T

    # Init variables
    n_times, n_m = A.shape
    V = np.eye(n_times)
    epsilon = n_times * (n_times - 1) * eps

    for it in range(n_iter_max):
        decr = 0
        for ii in range(1, n_times):
            for jj in range(ii):
                Ii = np.arange(ii, n_m, n_times)
                Ij = np.arange(jj, n_m, n_times)

                c1 = A[ii, Ii]
                c2 = A[jj, Ij]

                g12 = np.mean(A[ii, Ij] / c1)
                g21 = np.mean(A[ii, Ij] / c2)

                omega21 = np.mean(c1 / c2)
                omega12 = np.mean(c2 / c1)
                omega = np.sqrt(omega12 * omega21)

                tmp = np.sqrt(omega21 / omega12)
                tmp1 = (tmp * g12 + g21) / (omega + 1)
                tmp2 = (tmp * g12 - g21) / max(omega - 1, 1e-9)

                h12 = tmp1 + tmp2
                h21 = np.conj((tmp1 - tmp2) / tmp)

                decr += n_epochs * (g12 * np.conj(h12) + g21 * h21) / 2.0

                tmp = 1 + 1.j * 0.5 * np.imag(h12 * h21)
                tmp = np.real(tmp + np.sqrt(tmp ** 2 - h12 * h21))
                tau = np.array([[1, -h12 / tmp], [-h21 / tmp, 1]])

                A[[ii, jj], :] = np.dot(tau, A[[ii, jj], :])
                tmp = np.c_[A[:, Ii], A[:, Ij]]
                tmp = np.reshape(tmp, (n_times * n_epochs, 2), order='F')
                tmp = np.dot(tmp, tau.T)

                tmp = np.reshape(tmp, (n_times, n_epochs * 2), order='F')
                A[:, Ii] = tmp[:, :n_epochs]
                A[:, Ij] = tmp[:, n_epochs:]
                V[[ii, jj], :] = np.dot(tau, V[[ii, jj], :])
        if decr < epsilon:
            break
    D = np.reshape(A, (n_times, -1, n_times)).transpose(1, 0, 2)
    return V.T, D

def _uwedge(X, init=None, eps=1e-9, n_iter_max=1000):
    """Approximate joint diagonalization algorithm UWEDGE.

    Parameters
    ----------
    X : (n_trials, n_channels, n_channels) array_like
        A set of covariance matrices to diagonalize.
    init : (n_channels, n_channels) array_like, optional
        Initialization for the diagonalizer.
    eps : float, default 1e-9
        Tolerance for stoping criterion.
    n_iter_max : int, default 1000
        The maximum number of iteration to reach convergence.

    Returns
    -------
    W_est : (n_channels, n_filters) array_like
        The diagonalizer, usually n_filters == n_channels.
    D : (n_trials, n_channels, n_channels) array_like
        The set of quasi diagonal matrices.

    Notes
    -----
    Uniformly Weighted Exhaustive Diagonalization using Gauss iteration
    (U-WEDGE). Implementation of the AJD algorithm by Tichavsky and Yeredor [1]_ [2]_.
    This is a translation from the matlab code provided by the authors.

    References
    ----------
    .. [1] P. Tichavsky, A. Yeredor and J. Nielsen, "A Fast Approximate Joint Diagonalization Algorithm Using a Criterion with a Block Diagonal Weight Matrix", ICASSP 2008, Las Vegas.
    .. [2] P. Tichavsky and A. Yeredor, "Fast Approximate Joint Diagonalization Incorporating Weight Matrices" IEEE Transactions of Signal Processing, 2009.
    """
    L, d, _ = X.shape

    # reshape input matrix
    M = np.concatenate(X, 0).T

    # init variables
    d, Md = M.shape
    iteration = 0
    improve = 10

    if init is None:
        E, H = np.linalg.eig(M[:, 0:d])
        W_est = np.dot(np.diag(1. / np.sqrt(np.abs(E))), H.T)
    else:
        W_est = init

    Ms = np.array(M)
    Rs = np.zeros((d, L))

    for k in range(L):
        ini = k*d
        Il = np.arange(ini, ini + d)
        M[:, Il] = 0.5*(M[:, Il] + M[:, Il].T)
        Ms[:, Il] = np.dot(np.dot(W_est, M[:, Il]), W_est.T)
        Rs[:, k] = np.diag(Ms[:, Il])

    crit = np.sum(Ms**2) - np.sum(Rs**2)
    while (improve > eps) & (iteration < n_iter_max):
        B = np.dot(Rs, Rs.T)
        C1 = np.zeros((d, d))
        for i in range(d):
            C1[:, i] = np.sum(Ms[:, i:Md:d]*Rs, axis=1)

        D0 = B*B.T - np.outer(np.diag(B), np.diag(B))
        A0 = (C1 * B - np.dot(np.diag(np.diag(B)), C1.T)) / (D0 + np.eye(d))
        A0 += np.eye(d)
        W_est = np.linalg.solve(A0, W_est)

        Raux = np.dot(np.dot(W_est, M[:, 0:d]), W_est.T)
        aux = 1./np.sqrt(np.abs(np.diag(Raux)))
        W_est = np.dot(np.diag(aux), W_est)

        for k in range(L):
            ini = k*d
            Il = np.arange(ini, ini + d)
            Ms[:, Il] = np.dot(np.dot(W_est, M[:, Il]), W_est.T)
            Rs[:, k] = np.diag(Ms[:, Il])

        crit_new = np.sum(Ms**2) - np.sum(Rs**2)
        improve = np.abs(crit_new - crit)
        crit = crit_new
        iteration += 1

    D = np.reshape(Ms, (d, L, d)).transpose(1, 0, 2)
    return W_est.T, D

_ajd_methods = {
    'rjd': _rjd, 
    'ajd_pham': _ajd_pham, 
    'uwedge': _uwedge
}

def _check_ajd_method(method):
    """Check if a given approximate joint diagonalization method is valid.

    Parameters
    ----------
    method : callable object or str
        Could be the name of ajd_method or a callable method itself.

    Returns
    -------
    method: callable object
        A callable ajd method.
    """
    if callable(method):
        pass
    elif method in _ajd_methods.keys():
        method = _ajd_methods[method]
    else:
        raise ValueError(
            """%s is not an valid method ! Valid methods are : %s or a
             callable function""" % (method, (' , ').join(_ajd_methods.keys())))
    return method

def ajd(X, method='uwedge'):
    """Wrapper of approximate joint diagonalization methods.

    Parameters
    ----------
    X : (n_trials, n_channels, n_channels) array_like
        Input covariance matrices.
    method : str, default 'uwedge'
        The ajd method.
    
    Returns
    -------
    D : (n_channels,) array_like
        The mean of quasi diagonal matrices.
    W : (n_channels, n_filters) array_like
        The diagonalizer, usually n_filters == n_channels.
    """
    method = _check_ajd_method(method)
    W, D = method(X)
    D = np.diag(np.mean(D, axis=0))
    idx = np.argsort(D)[::-1]
    D = D[idx]
    W = W[:, idx]
    return D, W

def multicsp_kernel(X, y, cov_estimator='cov', ajd_method='ajd_pham'):
    r"""Grosse-Wentrup Multiclass CSP.

    Parameters
    ----------
    X : (n_trials, n_channels, n_samples) array_like
        EEG signals.
    y : (n_trials,) array_like
        Labels.
    cov_estimator : str, default 'cov'
        Covariance estimator to use.
    ajd_method : str, default 'ajd_pham'
        ajd method, 'uwedge' 'rjd' and 'ajd_pham'.

    Returns
    -------
    mutual_info : (n_channels,) array_like
        Mutual information values in descending order.
    W : (n_channels, n_channels) array_like
        Eigenvectors corresponding to mutual information values.

    References
    ----------
    .. [1] Grosse-Wentrup, Moritz, and Martin Buss. "Multiclass common spatial patterns and information theoretic feature extraction." Biomedical Engineering, IEEE Transactions on 55, no. 8 (2008): 1991-2000.
    """
    labels = np.unique(y)
    C = covariances(X, estimator=cov_estimator, n_jobs=1)
    C = C / np.trace(C, axis1=-1, axis2=-2)[:, np.newaxis, np.newaxis]
    Cx = np.stack([np.mean(C[y==label], axis=0) for label in labels], axis=0)
    D, W = ajd(Cx, method=ajd_method)
    W = W / np.sqrt(D)

    # compute mutual information values
    Pc = [np.mean(y == label) for label in labels]
    mutual_info = []
    for j in range(W.shape[-1]):
        a = 0
        b = 0
        for i in range(len(labels)):
            tmp = W[:, j].T@Cx[i]@W[:, j]
            a += Pc[i] * np.log(np.sqrt(tmp))
            b += Pc[i] * (tmp ** 2 - 1)
        mi = - (a + (3.0 / 16) * (b ** 2))
        mutual_info.append(mi)
    mutual_info = np.array(mutual_info)
    idx = np.argsort(mutual_info)[::-1]
    W = W[:, idx]
    mutual_info = mutual_info[idx]
    D = D[idx]
    return mutual_info, W

class MultiCSP(BaseEstimator, TransformerMixin, ClassifierMixin):
    r"""Multiclass common spatial pattern.

    Parameters
    ----------
    n_components : int, default 2
        The number of components.
    cov_estimator : str, default 'cov'
        Covariance estimator to use.
    ajd_method : str, default 'ajd_pham'
        ajd method, 'uwedge' 'rjd' and 'ajd_pham'.
    classifier : estimator, optional
        Classifier in sklearn.
        If None, skips classification step.

    Notes
    -----
    CSP [1]_ method for MI-BCI.

    References
    ----------
    .. [1] Grosse-Wentrup, Moritz, and Martin Buss. "Multiclass common spatial patterns and information theoretic feature extraction." Biomedical Engineering, IEEE Transactions on 55, no. 8 (2008): 1991-2000.
    """
    def __init__(self,
            n_components=2,
            cov_estimator='cov',
            ajd_method='uwedge',
            classifier=None):
        self.n_components = n_components
        self.cov_estimator = cov_estimator
        self.ajd_method = ajd_method
        self.classifier = classifier
        self.classifier_ = None

    def fit(self, X, y):
        r"""Fit to data.

        Parameters
        ----------
        X : (n_trials, n_channels, n_samples) array_like
            EEG signal.
        y : (n_trials,) array_like
            Labels.
        """
        X = np.reshape(X, (-1, *X.shape[-2:]))
        _, W = multicsp_kernel(
            X, y, cov_estimator=self.cov_estimator, ajd_method=self.ajd_method)
        self.W_ = W

        if self.classifier is not None:
            self.classifier_ = clone(self.classifier)
            feat = csp_feature(
                X, W, n_components=self.n_components)
            self.classifier_.fit(feat, y)
        return self

    def transform(self, X):
        r"""Transform data.

        Parameters
        ----------
        X : (n_trials, n_channels, n_samples) array_like
            EEG signal.

        Returns
        -------
        (n_trials, n_components) array_like
            CSP features.
        """
        X = np.reshape(X, (-1, *X.shape[-2:]))
        features = csp_feature(X, self.W_, n_components=self.n_components)
        return features

    def predict(self, X):
        r"""Predict data.

        Parameters
        ----------
        X : (n_trials, n_channels, n_samples) array_like
            EEG signal.

        Returns
        -------
        labels : (n_trials,) array_like
            Class labels.
        """
        feat = self.transform(X)
        if self.classifier_ is not None:
            labels = self.classifier_.predict(feat)
            return labels
        else:
            raise NotImplementedError("No classifier was provided to support predict function.")

class FBMultiCSP(FilterBank, ClassifierMixin):
    r"""Filterbank MultiCSP.

    Parameters
    ----------
    filterbank : list
        List of filter coefficients in sos form.
    n_components : int, default 1
        The number of components.
    cov_estimator : str, default 'cov'
        Covariance estimator to use.
    ajd_method : str, default 'ajd_pham'
        ajd method, 'uwedge' 'rjd' and 'ajd_pham'.
    classifier : estimator, optional
        Classifier in sklearn.
        If None, skips classification step.
    n_mi_components : int, default 1
        The number of mutual information selection components.
    n_jobs : int, optional
        The number of CPUs to use to do the computation.

    Notes
    -----
    This implementation uses mutual information to select features, as described in [1]_.

    References
    ----------
    .. [1] Ang K K, Chin Z Y, Zhang H, et al. Filter bank common spatial pattern (FBCSP) in brain-computer interface[C]//2008 IEEE International Joint Conference on Neural Networks (IEEE World Congress on Computational Intelligence). IEEE, 2008: 2390-2397.
    .. [2] Grosse-Wentrup, Moritz, and Martin Buss. "Multiclass common spatial patterns and information theoretic feature extraction." Biomedical Engineering, IEEE Transactions on 55, no. 8 (2008): 1991-2000.
    """
    def __init__(self,
        filterbank,
        n_components=2,
        cov_estimator='cov',
        ajd_method='uwedge',
        classifier=None,
        n_mi_components=4,
        n_jobs=None):
        self.filterbank = filterbank
        self.n_components = n_components
        self.n_mi_components = n_mi_components
        self.cov_estimator = cov_estimator
        self.ajd_method = ajd_method
        self.classifier = classifier
        self.classifier_ = None
        self.n_jobs = n_jobs

        super().__init__(
            MultiCSP(
                n_components=n_components,
                cov_estimator=cov_estimator,
                ajd_method=ajd_method,
                classifier=None),
            filterbank,
            concat=True,
            n_jobs=n_jobs)

    def fit(self, X, y):
        r"""Fit to data.

        Parameters
        ----------
        X : (n_trials, n_channels, n_samples) array_like
            EEG signal.
        y : (n_trials,) array_like
            Labels.
        """
        super().fit(X, y, axis=-1)

        # feature selection step
        feat = super().transform(X)
        self.selector_ = SelectKBest(
            score_func=mutual_info_classif,
            k=self.n_mi_components)
        self.selector_.fit(feat, y)

        if self.classifier is not None:
            feat = self.selector_.transform(feat)
            self.classifier_ = clone(self.classifier)
            self.classifier_.fit(feat, y)
        return self

    def transform(self, X):
        r"""Transform data.

        Parameters
        ----------
        X : (n_trials, n_channels, n_samples) array_like
            EEG signal.

        Returns
        -------
        (n_trials, n_mi_components) array_like
            CSP features with mutual information selection.
        """
        features = super().transform(X, axis=-1)
        features = self.selector_.transform(features)
        return features

    def predict(self, X):
        r"""Predict data.

        Parameters
        ----------
        X : (n_trials, n_channels, n_samples) array_like
            EEG signal.

        Returns
        -------
        labels : (n_trials,) array_like
            Class labels.
        """
        feat = self.transform(X)
        if self.classifier_ is not None:
            labels = self.classifier_.predict(feat)
            return labels
        else:
            raise NotImplementedError("No classifier was provided to support predict function.")

