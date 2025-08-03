# -*- coding: utf-8 -*-
# Authors: swolf <swolfforever@gmail.com>
# Date: 2023/01/02
# License: MIT License
import numpy as np
from scipy.signal import sosfiltfilt, cheby1, cheb1ord
from sklearn.base import BaseEstimator, TransformerMixin, clone
from joblib import Parallel, delayed

__all__ = [
    "pearsonr",
    "generate_filterbank",
    "FilterBank",
    "generate_ssvep_reference",
]


def pearsonr(X, Y=None):
    r"""Pearson's correlation coefficient.

    Parameters
    ----------
    X : (n_samples, n_features) array_like
        The first input array.
    Y : (n_samples, n_features) array_like, optional
        The second input array.
        If None, Pearson's correlation coefficient computed on X itself.

    Returns
    -------
    r : (n_samples,) array_like
        Pearson correlation coefficients.
    """
    eps = np.finfo(X.dtype).eps
    if Y is None:
        Y = X
    X = np.reshape(X, (-1, X.shape[-1]))
    Y = np.reshape(Y, (-1, Y.shape[-1]))
    m = X.shape[1]
    if m != Y.shape[1]:
        raise ValueError("X and Y must have the same shape.")

    X = X - np.mean(X, axis=-1, keepdims=True)
    Y = Y - np.mean(Y, axis=-1, keepdims=True)
    normx = np.linalg.norm(X, axis=-1, keepdims=True)
    normy = np.linalg.norm(Y, axis=-1, keepdims=True)

    X = X / (normx + eps)
    Y = Y / (normy + eps)

    r = np.sum(X * Y, axis=-1)
    return r


def generate_filterbank(passbands, stopbands, srate, order=None, rp=0.5):
    r"""Generate filter bank with cheby1.

    Parameters
    ----------
    passbands : list
        List of passband frequencies.
    stopbands : list
        List of stopband frequencies.
    srate : int or float
        The sampling frequency.
    order : int, optional
        The order of the cheby1 filter.
        If None, the order is induced from the cheb1ord function.
    rp : float, default 0.5
        The maximum ripple allowed below unity gain in the passband.
        Specified in decibels, as a positive number.

    Returns
    -------
    filterbanks : list
        List of filter coefficients in sos form.
    """
    filterbanks = []
    for wp, ws in zip(passbands, stopbands):
        if order is None:
            N, wn = cheb1ord(wp, ws, 3, 40, fs=srate)
            sos = cheby1(N, rp, wn, btype="bandpass", output="sos", fs=srate)
        else:
            sos = cheby1(order, rp, wp, btype="bandpass", output="sos", fs=srate)
        filterbanks.append(sos)
    return filterbanks


class FilterBank(BaseEstimator, TransformerMixin):
    r"""Basic filterbank design.

    Parameters
    ----------
    base_estimator : estimator
        The base estimator object.
    filterbank : list
        List of filter coefficients in sos form.
    concat : bool, default False
        Return concated features, else return stacked features.
    n_jobs : int, optional
        The number of CPUs to use to do the computation.
    """

    def __init__(self, base_estimator, filterbank, concat=False, n_jobs=None):
        self.base_estimator = base_estimator
        self.filterbank = filterbank
        self.concat = concat
        self.n_jobs = n_jobs

    def fit(self, X, y=None, axis=-1, **kwargs):
        r"""Fit to data.

        Parameters
        ----------
        X : array_like
            Input samples.
        y : array_like, optional
            Labels, None for unsupervised fit.
        axis : int, default -1
            The axis of X to which the filter is applied.
        **kwargs : dict
            Additional fit parameters.
        """
        self.estimators_ = [
            clone(self.base_estimator) for _ in range(len(self.filterbank))
        ]

        def wrapper(sos_coef, est, X, y, axis, kwargs):
            X = sosfiltfilt(sos_coef, X, axis=axis)
            est.fit(X, y, **kwargs)
            return est

        self.estimators_ = Parallel(n_jobs=self.n_jobs)(
            delayed(wrapper)(sos_coef, est, X, y, axis, kwargs)
            for sos_coef, est in zip(self.filterbank, self.estimators_)
        )
        return self

    def transform(self, X, axis=-1, **kwargs):
        r"""Transform data.

        Parameters
        ----------
        X : array_like
            Input samples.
        axis : int, default -1
            The axis of X to which the filter is applied.
        **kwargs : dict
            Additional transform parameters.
        """

        def wrapper(sos_coef, est, X, axis, kwargs):
            X = sosfiltfilt(sos_coef, X, axis=axis)
            retval = est.transform(X, **kwargs)
            return retval

        features = Parallel(n_jobs=self.n_jobs)(
            delayed(wrapper)(sos_coef, est, X, axis, kwargs)
            for sos_coef, est in zip(self.filterbank, self.estimators_)
        )
        if self.concat:
            features = np.concatenate(features, axis=-1)
        else:
            features = np.stack(features, axis=-1)
        return features


def generate_ssvep_reference(freqs, srate, T, phases=None, n_harmonics=1):
    r"""Generate SSVEP reference signals.

    Parameters
    ----------
    freqs : (n_freqs,) array_like
        The stimulus frequencies.
    srate : float or int
        The sampling rate of the monitor.
    T : float
        The time for stimulus in seconds.
    phases : (n_freqs,) array_like, optional
        The stimulus phases, default 0.
    n_harmonics : int, default 1
        The number of harmonics.

    Returns
    -------
    Yf : (n_freqs, 2*n_harmonics, int(T*srate)) array_like
        Reference signals
    """
    if isinstance(freqs, int) or isinstance(freqs, float):
        freqs = [freqs]
    freqs = np.array(freqs)[:, np.newaxis]
    if phases is None:
        phases = 0
    if isinstance(phases, int) or isinstance(phases, float):
        phases = [phases]
    phases = np.array(phases)[:, np.newaxis]
    t = np.linspace(0, T, int(T * srate))

    Yf = []
    for i in range(n_harmonics):
        Yf.append(
            np.stack(
                [
                    np.sin(2 * np.pi * (i + 1) * freqs * t + np.pi * phases),
                    np.cos(2 * np.pi * (i + 1) * freqs * t + np.pi * phases),
                ],
                axis=1,
            )
        )
    Yf = np.concatenate(Yf, axis=1)
    return Yf
