# -*- coding: utf-8 -*-
#
# Authors: Swolf <swolfforever@gmail.com>
# Date: 2023/01/20
# License: MIT License
"""Riemannian geometry methods.
"""
import numpy as np
from scipy.linalg import eigvalsh, pinv
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin, clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from psygine.decoders.utils import covariances, sqrtm, invsqrtm, logm, expm, powm

__all__ = [
    "logmap",
    "expmap",
    "geodesic",
    "distance_riemann",
    "mean_riemann",
    "vectorize",
    "unvectorize",
    "tangent_space",
    "untangent_space",
    "mdrm_kernel",
    "MDRM",
    "FGDA",
    "FgMDRM",
    "TSClassifier",
    "Align",
    "AdaAlign",
]


def logmap(Pi, P, n_jobs=None):
    r"""Logarithm map from the positive-definite space to the tangent space.

    Parameters
    ----------
    Pi : (n_trials, n_channels, n_channels) array_like
        SPD matrices.
    P : (n_trials, n_channels, n_channels) or (n_channels, n_channels) array_like
        Reference points.
        If a group of points are provided, the shape of P should be the same as the shape of Pi.
    n_jobs : int, optional
        The number of cpu cores to use.

    Returns
    -------
    Si : (n_trials, n_channels, n_channels) array_like
        Tangent space points.

    Notes
    -----
    Logarithm map projects :math:`\mathbf{P}_i \in \mathcal{M}` to the tangent space point
    :math:`\mathbf{S}_i \in \mathcal{T}_{\mathbf{P}} \mathcal{M}` at :math:`\mathbf{P} \in \mathcal{M}`.
    """
    P12 = sqrtm(P, n_jobs=n_jobs)
    iP12 = invsqrtm(P, n_jobs=n_jobs)
    wPi = np.matmul(np.matmul(iP12, Pi), iP12)
    Si = np.matmul(np.matmul(P12, logm(wPi, n_jobs=n_jobs)), P12)
    return Si


def expmap(Si, P, n_jobs=None):
    r"""Exponential map from the tangent space to the positive-definite space.

    Parameters
    ----------
    Si : (n_trials, n_channels, n_channels) array_like
        Tangent space points.
    P : (n_trials, n_channels, n_channels) or (n_channels, n_channels) array_like
        Reference points.
        If a group of points are provided, the shape of P should be the same as the shape of Si.
    n_jobs : int, optional
        The number of cpu cores to use.

    Returns
    -------
    Pi : (n_trials, n_channels, n_channels) array_like
        SPD matrices.

    Notes
    -----
    Exponential map projects :math:`\mathbf{S}_i \in \mathcal{T}_{\mathbf{P}} \mathcal{M}` bach to the manifold :math:`\mathcal{M}`.
    """
    P12 = sqrtm(P, n_jobs=n_jobs)
    iP12 = invsqrtm(P, n_jobs=n_jobs)
    wSi = np.matmul(np.matmul(iP12, Si), iP12)
    Pi = np.matmul(np.matmul(P12, expm(wSi, n_jobs=n_jobs)), P12)
    return Pi


def geodesic(P1, P2, t, n_jobs=None):
    r"""Geodesic between P1 and P2.

    Parameters
    ----------
    P1 : (n_trials, n_channels, n_channels) array_like
        SPD matrices.
    P2 : (n_trials, n_channels, n_channels) array_like
        SPD matrices, the same shape of P1.
    t : float
        The control point ranges from 0 to 1.
    n_jobs : int, optional
        The number of cpu cores to use.

    Returns
    -------
    phi : (n_trials, n_channels, n_channels) array_like
        SPD matrices on the geodesic curve between P1 and P2.

    Notes
    -----
    The geodesic curve between any two SPD matrices :math:`\mathbf{P}_1,\mathbf{P}_2 \in \mathcal{M}`.
    """
    p1_shape = P1.shape
    p2_shape = P2.shape
    P1 = P1.reshape((-1, *p1_shape[-2:]))
    P2 = P2.reshape((-1, *p2_shape[-2:]))
    P12 = sqrtm(P1, n_jobs=n_jobs)
    iP12 = invsqrtm(P1, n_jobs=n_jobs)
    wP2 = np.matmul(np.matmul(iP12, P2), iP12)
    phi = np.matmul(np.matmul(P12, powm(wP2, t, n_jobs=n_jobs)), P12)
    return phi


def distance_riemann(A, B, n_jobs=None):
    r"""Riemannian distance between two covariance matrices A and B.

    Parameters
    ----------
    A : (n_trials, n_channels, n_channels) array_like
        The first positive-definite matrix.
    B : (n_trials, n_channels, n_channels) array_like
        The second positive-definite matrix.
    n_jobs : int, optional
        The number of cpu cores to use.

    Returns
    -------
    (n_trials,) array_like
        Riemannian distance between A and B.

    Notes
    -----
    .. math::
        d = {\left( \sum_i \log(\lambda_i)^2 \\right)}^{-1/2}

    where :math:`\lambda_i` are the joint eigenvalues of A and B.
    """

    def _single_distance_riemann(A, B):
        dist = np.sqrt(np.sum(np.log(eigvalsh(A, B)) ** 2))
        return dist

    A = A.reshape((-1, *A.shape[-2:]))
    B = B.reshape((-1, *B.shape[-2:]))

    if A.shape[0] == 1:
        A = np.broadcast_to(A, B.shape)
    elif B.shape[0] == 1:
        B = np.broadcast_to(B, A.shape)

    d = Parallel(n_jobs=n_jobs)(
        delayed(_single_distance_riemann)(a, b) for a, b in zip(A, B)
    )
    d = np.array(d)
    return d


def _get_sample_weight(sample_weight, N):
    r"""Get the sample weights.

    If none provided, weights are initialized to 1.
    """
    if sample_weight is None:
        sample_weight = np.ones(N)
    if len(sample_weight) != N:
        raise ValueError("len of sample_weight must be equal to len of data.")
    return sample_weight


def mean_riemann(covmats, tol=1e-11, maxiter=300, init=None, weights=None, n_jobs=None):
    r"""Return the mean covariance matrix according to the Riemannian metric.

    Parameters
    ----------
    covmats : (n_trials, n_channels, n_channels) array_like
        Covariance matrices.
    tol : float, default 1e-11
        The tolerance to stop the gradient descent.
    maxiter : int, default 300
        The maximum number of iteration.
    init : (n_channels, n_channels) array_like, optional
        A covariance matrix used to initialize the gradient descent, if None the arithmetic mean is used.
    weights : (n_trials,) array_like, optional
        The weight of each sample, if None weights are initialized to 1.
    n_jobs : int, optional
        The number of cpu cores to use.

    Returns
    -------
    C : (n_channels, n_channels) array_like
        The Riemannian mean covariance matrix.

    Notes
    -----
    The procedure is similar to a gradient descent minimizing the sum of riemannian distance to the mean.

    .. math::
        \mathbf{C} = \\arg \min{(\sum_i \delta_R ( \mathbf{C} , \mathbf{C}_i)^2)}

    where :math:\delta_R is riemann distance.
    """
    covmats = np.reshape(covmats, (-1, *covmats.shape[-2:]))
    # init
    weights = _get_sample_weight(weights, len(covmats))
    if init is None:
        C = np.mean(covmats, axis=0)
    else:
        C = init
    k = 0
    nu = 1.0
    tau = np.finfo(np.float64).max
    crit = np.finfo(np.float64).max
    # stop when J<10^-9 or max iteration = 50
    while (crit > tol) and (k < maxiter) and (nu > tol):
        k = k + 1
        C12 = sqrtm(C, n_jobs=1)
        iC12 = invsqrtm(C, n_jobs=1)

        J = logm(np.matmul(np.matmul(iC12, covmats), iC12), n_jobs=n_jobs)
        J = np.sum(weights[:, np.newaxis, np.newaxis] * J, axis=0)
        crit = np.linalg.norm(J, ord="fro")
        h = nu * crit

        C = C12 @ expm(nu * J, n_jobs=1) @ C12
        if h < tau:
            nu = 0.95 * nu
            tau = h
        else:
            nu = 0.5 * nu
    return C


def vectorize(Si):
    r"""vectorize tangent space points.

    Parameters
    ----------
    Si : (n_trials, n_channels, n_channels) array_like
        Points in the tangent space.

    Returns
    -------
    (n_trials, n_channels*(n_channels+1)/2)
        Vectorized version of Si.
    """
    Si = Si.reshape((-1, *Si.shape[-2:]))
    n_channels = Si.shape[-1]
    ind = np.triu_indices(n_channels, k=0)
    coeffs = (
        np.sqrt(2) * np.triu(np.ones((n_channels, n_channels)), 1) + np.eye(n_channels)
    )[ind]
    vSi = Si[:, ind[0], ind[1]] * coeffs
    return vSi


def unvectorize(vSi):
    r"""Unvectorize tangent space points.

    Parameters
    ----------
    vSi : (n_trials, n_channels*(n_channels+1)/2) array_like
        Vectorized version of Si.

    Returns
    -------
    (n_trials, n_channels, n_channels) array_like
        Points in the tangent space.
    """
    n_trials, n_features = vSi.shape
    n_channels = int((np.sqrt(1 + 8 * n_features) - 1) / 2)
    ind = np.triu_indices(n_channels, k=0)
    coeffs = (
        np.sqrt(2) * np.triu(np.ones((n_channels, n_channels)), 1)
        + 2 * np.eye(n_channels)
    )[ind]
    vSi = vSi / coeffs
    Si = np.zeros((n_trials, n_channels, n_channels))
    Si[:, ind[0], ind[1]] = vSi
    Si = Si + np.transpose(Si, (0, 2, 1))
    return Si


def tangent_space(Pi, P, n_jobs=None):
    r"""Logarithm map projects SPD matrices to the tangent vectors.

    Parameters
    ----------
    Pi : (n_trials, n_channels, n_channels) array_like
        SPD matrices.
    P : (n_trials, n_channels, n_channels) or (n_channels, n_channels) array_like
        Reference points.
        If a group of points are provided, the shape of P should be the same as the shape of Pi.
    n_jobs : int, optional
        The number of cpu cores to use.

    Returns
    -------
    vSi : (n_trials, n_channels*(n_channels+1)/2) array_like
        Tangent vectors in vectorized form.
    """
    Si = logmap(Pi, P, n_jobs=n_jobs)
    vSi = vectorize(Si)
    return vSi


def untangent_space(vSi, P, n_jobs=None):
    r"""Logarithm map projects SPD matrices to the tangent vectors.

    Parameters
    ----------
    vSi : (n_trials, n_channels*(n_channels+1)/2) array_like
        Tangent vectors in vecotrized form.
    P : (n_trials, n_channels, n_channels) or (n_channels, n_channels) array_like
        Reference points.
        If a group of points are provided, the shape of P should be the same as the shape of Pi.
    n_jobs : int, optional
        The number of cpu cores to use.

    Returns
    -------
    Pi : (n_trials, n_channels, n_channels) array_like
        SPD matrices.
    """
    Si = unvectorize(vSi)
    Pi = expmap(Si, P, n_jobs=n_jobs)
    return Pi


def mdrm_kernel(X, y, cov_estimator="cov", weights=None, n_jobs=None):
    r"""Minimum Distance to Riemannian Mean.

    Parameters
    ----------
    X : (n_trials, n_channels, n_samples) array_like
        EEG signals.
    y : (n_trials,) array_like
        Labels.
    cov_estimator : str, default 'cov'
        Covariance estimator to use.
    weights : (n_trials,) array_like, optional
        Sample weights, if None weights are initialized to 1.
    n_jobs : int, optional
        The number of cpu cores to use.

    Returns
    -------
    (n_class, n_channels, n_channels) array_like
        Centroids of each class.
    """
    labels = np.unique(y)
    Cx = covariances(X, estimator=cov_estimator, n_jobs=1)
    weights = _get_sample_weight(weights, Cx.shape[0])

    Centroids = Parallel(n_jobs=n_jobs)(
        delayed(mean_riemann)(Cx[y == label], sample_weight=weights[y == label])
        for label in labels
    )
    return np.stack(Centroids)


class MDRM(BaseEstimator, TransformerMixin, ClassifierMixin):
    def __init__(self, cov_estimator="cov", n_jobs=None):
        self.cov_estimator = cov_estimator
        self.n_jobs = n_jobs

    def fit(self, X, y, weights=None):
        self.classes_ = np.unique(y)
        self.centroids_ = mdrm_kernel(
            X, y, cov_estimator=self.cov_estimator, weights=weights, n_jobs=self.n_jobs
        )
        return self

    def transform(self, X):
        Cx = covariances(X, estimator=self.cov_estimator, n_jobs=1)
        dist = np.stack(
            [
                distance_riemann(Cx, centroid, n_jobs=self.n_jobs)
                for centroid in self.centroids_
            ]
        ).T
        return dist

    def predict(self, X):
        dist = self.transform(X)
        return self.classes_[np.argmin(dist, axis=-1)]


class FGDA(BaseEstimator, TransformerMixin):
    def __init__(self, cov_estimator="cov", n_jobs=None):
        self.cov_estimator = cov_estimator
        self.n_jobs = n_jobs

    def fit(self, X, y):
        Pi = covariances(X, estimator=self.cov_estimator, n_jobs=1)
        self.P_ = mean_riemann(Pi, n_jobs=self.n_jobs)
        vSi = tangent_space(Pi, self.P_, n_jobs=self.n_jobs)
        self.lda_ = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
        self.lda_.fit(vSi, y)
        W = self.lda_.coef_.copy()
        self.W_ = W.T @ pinv(W @ W.T) @ W  # n_feat by n_feat
        return self

    def transform(self, X):
        Pi = covariances(X, estimator=self.cov_estimator, n_jobs=1)
        vSi = tangent_space(Pi, self.P_, n_jobs=self.n_jobs)
        vSi = vSi @ self.W_
        Pi = untangent_space(vSi, self.P_, n_jobs=self.n_jobs)
        return Pi


class FgMDRM(BaseEstimator, TransformerMixin, ClassifierMixin):
    def __init__(self, cov_estimator="cov", n_jobs=None):
        self.cov_estimator = cov_estimator
        self.n_jobs = n_jobs

    def fit(self, X, y, weights=None):
        self.classes_ = np.unique(y)
        self.fgda_ = FGDA(cov_estimator=self.cov_estimator, n_jobs=self.n_jobs)
        Cx = self.fgda_.fit_transform(X, y)
        weights = _get_sample_weight(weights, Cx.shape[0])
        Centroids = Parallel(n_jobs=self.n_jobs)(
            delayed(mean_riemann)(Cx[y == label], sample_weight=weights[y == label])
            for label in self.classes_
        )
        self.centroids_ = np.stack(Centroids)
        return self

    def transform(self, X):
        Cx = self.fgda_.transform(X)
        dist = np.stack(
            [
                distance_riemann(Cx, centroid, n_jobs=self.n_jobs)
                for centroid in self.centroids_
            ]
        ).T
        return dist

    def predict(self, X):
        dist = self.transform(X)
        return self.classes_[np.argmin(dist, axis=-1)]


class TSClassifier(BaseEstimator, TransformerMixin, ClassifierMixin):
    def __init__(self, cov_estimator="cov", classifier=None, n_jobs=None):
        self.cov_estimator = cov_estimator
        self.classifier = classifier
        self.classifier_ = None
        self.n_jobs = n_jobs

        if self.classifier is not None and not isinstance(
            self.classifier, ClassifierMixin
        ):
            raise TypeError("classifier must be a ClassifierMixin object")

    def fit(self, X, y):
        Pi = covariances(X, estimator=self.cov_estimator, n_jobs=self.n_jobs)
        self.P_ = mean_riemann(Pi, n_jobs=self.n_jobs)
        vSi = tangent_space(Pi, self.P_, n_jobs=self.n_jobs)

        if self.classifier is not None:
            self.classifier_ = clone(self.classifier)
            self.classifier_.fit(vSi, y)
        return self

    def transform(self, X):
        Pi = covariances(X, estimator=self.cov_estimator, n_jobs=self.n_jobs)
        vSi = tangent_space(Pi, self.P_, n_jobs=self.n_jobs)
        return vSi

    def predict(self, X):
        vSi = self.transform(X)
        if self.classifier_ is not None:
            labels = self.classifier_.predict(vSi)
            return labels
        else:
            raise NotImplementedError(
                "No classifier was provided to support predict function."
            )


class Align(BaseEstimator, TransformerMixin):
    def __init__(self, cov_estimator="cov", align_method="euclid", n_jobs=None):
        self.cov_estimator = cov_estimator
        self.align_method = align_method
        self.n_jobs = n_jobs

    def fit(self, X, y=None):
        Cs = covariances(X, estimator=self.cov_estimator, n_jobs=1)
        if self.align_method == "euclid":
            self.iC12_ = self._euclid_center(Cs)
        elif self.align_method == "riemann":
            self.iC12_ = self._riemann_center(Cs)
        else:
            raise ValueError("non-supported aligning method.")
        return self

    def transform(self, X):
        X = np.reshape(X, (-1, *X.shape[-2:]))
        X = X - np.mean(X, axis=-1, keepdims=True)
        X = np.matmul(self.iC12_, X)
        return X

    def _euclid_center(self, Cs):
        C = np.mean(Cs, axis=0)
        return invsqrtm(C)

    def _riemann_center(self, Cs):
        C = mean_riemann(Cs, n_jobs=self.n_jobs)
        return invsqrtm(C)


class AdaAlign(BaseEstimator, TransformerMixin):
    def __init__(self, cov_estimator="cov", align_method="euclid"):
        self.cov_estimator = cov_estimator
        self.align_method = align_method
        self.C_ = None
        self.iC12_ = None
        self.n_tracked_ = None

    def fit(self, X=None, y=None):
        return self

    def transform(self, X):
        X = np.reshape(X, (-1, *X.shape[-2:]))
        X = X - np.mean(X, axis=-1, keepdims=True)
        Cs = covariances(X, estimator=self.cov_estimator, n_jobs=1)
        X = self._recursive_fit_transform(X, Cs)
        return X

    def _recursive_fit_transform(self, X, Cs):
        self._init_center(Cs.shape[-1])

        for i in range(len(X)):
            if self.align_method == "euclid":
                self._recursive_euclid_center(Cs[i])
            elif self.align_method == "riemann":
                self._recursive_riemann_center(Cs[i])
            else:
                raise ValueError("non-supported aligning method.")
            X[i] = self.iC12_ @ X[i]
        return X

    def _init_center(self, n_channels):
        if self.iC12_ is None:
            self.iC12_ = np.eye(n_channels)
            self.C_ = np.eye(n_channels)
            self.n_tracked_ = 1

    def _recursive_euclid_center(self, C):
        if self.n_tracked_ == 1:
            self.C_ = self.C_ * np.mean(np.diag(C))

        self.n_tracked_ += 1
        alpha = 1 / (self.n_tracked_)
        self.C_ = (1 - alpha) * self.C_ + alpha * C
        self.iC12_ = invsqrtm(self.C_)

    def _recursive_riemann_center(self, C):
        if self.n_tracked_ == 1:
            self.C_ = self.C_ * np.mean(np.diag(C))

        self.n_tracked_ += 1
        alpha = 1 / (self.n_tracked_)
        self.C_ = geodesic(self.C_, C, alpha)
        self.iC12_ = invsqrtm(self.C_)
