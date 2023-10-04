# -*- coding: utf-8 -*-
# Authors: swolf <swolfforever@gmail.com>
# Date: 2023/01/11
# License: MIT License
"""Model selection methods.
"""
import warnings
import random
import numpy as np
from sklearn.model_selection import (StratifiedKFold, StratifiedShuffleSplit, LeaveOneGroupOut)

__all__ = [
    'set_random_seeds', 'EnhancedStratifiedKFold', 'EnhancedLeaveOneGroupOut', 'EnhancedStratifiedShuffleSplit'
]

def set_random_seeds(seed):
    r"""Set random seed for python, numpy and torch.

    Parameters
    ----------
    seed: int
        Random seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            # torch.cuda.manual_seed_all(seed)
            torch.cuda.manual_seed(seed)
            # Disable the inbuilt cudnn auto-tuner that finds the best algorithm to use for your hardware.
            torch.backends.cudnn.benchmark = False
            # Certain operations in Cudnn are not deterministic, and this line will force them to behave!
            torch.backends.cudnn.deterministic = True
    except ImportError:
        pass

class EnhancedStratifiedKFold(StratifiedKFold):
    r"""Wrapped Stratified K-Folds cross-validator.

    Provides train/validate/test indices to split data in train/validate/test sets.

    This cross-validation object is a variation of KFold that returns
    stratified folds. The folds are made by preserving the percentage of
    samples for each class.

    Parameters
    ----------
    n_splits : int, default 5
        Number of folds. Must be at least 2.
    shuffle : bool, default False
        Whether to shuffle each class's samples before splitting into batches.
        Note that the samples within each split will not be shuffled.
    validate_set : bool, default False
        Whether to return validate set.
    validate_size : float, default 0.1
        If float, should be between 0.0 and 1.0 and represent the proportion
        of the dataset to include in the validate split.
    random_state : int, optional
        When `shuffle` is True, `random_state` affects the ordering of the
        indices, which controls the randomness of each fold for each class.
    """
    def __init__(self,
        n_splits=5,
        shuffle=False,
        validate_set=False,
        validate_size=0.1,
        random_state=None):
        self.validate_set = validate_set
        if self.validate_set:
            if not isinstance(validate_size, float):
                raise ValueError("validate size should be float")
            self.validate_spliter = StratifiedShuffleSplit(
                n_splits=1, test_size=validate_size, random_state=random_state)
        super().__init__(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

    def split(self, X, y, groups=None):
        r"""Generate indices to split data into training, validate and test set.

        Parameters
        ----------
        X : (n_samples, n_features) array_like
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.
        y : (n_samples,) array_like
            The target variable for supervised learning problems.
            Stratification is done based on the y labels.
        groups : optional
            Always ignored, exists for compatibility.

        Yields
        ------
        train : ndarray
            The training set indices for that split.
        validate : ndarray
            The validating set indices for that split.
            Available only if validate_set is True.
        test : ndarray
            The testing set indices for that split.
        """
        for train, test in super().split(X, y, groups=groups):
            if self.validate_set:
                train_idx, validate_idx = next(self.validate_spliter.split(X[train], y[train], groups=groups))
                yield train[train_idx], train[validate_idx], test
            else:
                yield train, test

class EnhancedLeaveOneGroupOut(LeaveOneGroupOut):
    r"""Wrapped Leave One Group Out cross-validator.

    Provides train/validate/test indices to split data according to a third-party
    provided group. This group information can be used to encode arbitrary
    domain specific stratifications of the samples as integers.

    For instance the groups could be the year of collection of the samples
    and thus allow for cross-validation against time-based splits.

    Parameters
    ----------
    validate_set : bool, default False
        Whether to return validate set.
    """
    def __init__(self, validate_set=False):
        super().__init__()
        self.validate_set = validate_set
        if self.validate_set:
            self.validate_spliter = LeaveOneGroupOut()

    def split(self, X, y=None, groups=None):
        r"""Generate indices to split data into training, validating and test set.

        Parameters
        ----------
        X : (n_samples, n_features) array_like
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.
        y : (n_samples,) array_like
            The target variable for supervised learning problems.
            Stratification is done based on the y labels.
        groups : (n_samples,) array-like
            Group labels for the samples used while splitting the dataset into
            train/test set.

        Yields
        ------
        train : ndarray
            The training set indices for that split.
        validate : ndarray
            The validating set indices for that split.
            Available only if validate_set is True.
        test : ndarray
            The testing set indices for that split.
        """
        if groups is None and y is not None:
            groups = self._generate_sequential_groups(y)
        n_splits = super().get_n_splits(groups=groups)
        for train, test in super().split(X, y, groups):
            if self.validate_set:
                n_repeat = np.random.randint(1, n_splits)
                validate_iter = self.validate_spliter.split(X[train], y[train], groups[train])
                for _ in range(n_repeat):
                    train_idx, validate_idx = next(validate_iter)
                yield train[train_idx], train[validate_idx], test
            else:
                yield train, test

    def _generate_sequential_groups(self, y):
        labels = np.unique(y)
        groups = np.zeros((len(y)))
        idxs = [y==label for label in labels]
        n_labels = [np.sum(idx) for idx in idxs]
        if len(np.unique(n_labels)) > 1:
            warnings.warn("y is not balanced, the generated groups is not balanced as well.", RuntimeWarning)
        for idx, n_label in zip(idxs, n_labels):
            groups[idx] = np.arange(n_label)
        return groups

class EnhancedStratifiedShuffleSplit(StratifiedShuffleSplit):
    r"""Wrapped Stratified ShuffleSplit cross-validator.

    Provides train/validate/test indices to split data in train/validate/test sets.

    This cross-validation object is a merge of StratifiedKFold and
    ShuffleSplit, which returns stratified randomized folds. The folds
    are made by preserving the percentage of samples for each class.

    Note: like the ShuffleSplit strategy, stratified random splits
    do not guarantee that all folds will be different, although this is
    still very likely for sizeable datasets.

    Parameters
    ----------
    test_size : float
        If float, should be between 0.0 and 1.0 and represent the proportion
        of the dataset to include in the test split.
    n_splits : int, default 5
        Number of folds. Must be at least 2.
    shuffle : bool, default False
        Whether to shuffle each class's samples before splitting into batches.
        Note that the samples within each split will not be shuffled.
    validate_set : bool, default False
        Whether to return validate set.
    validate_size : float, default 0.1
        If float, should be between 0.0 and 1.0 and represent the proportion
        of the dataset to include in the validate split.
    random_state : int, optional
        When `shuffle` is True, `random_state` affects the ordering of the
        indices, which controls the randomness of each fold for each class.
    """
    def __init__(self,
        test_size,
        n_splits=5,
        validate_set=False,
        validate_size=0.1,
        random_state=None):
        self.validate_set = validate_set
        if not isinstance(test_size, float):
            raise ValueError("test size should be float")

        if self.validate_set:
            if not isinstance(validate_size, float):
                raise ValueError("validate size should be float")
            train_size = 1 - test_size - validate_size
        else:
            train_size = 1 - test_size
            validate_size = 0.0

        super().__init__(
            n_splits=n_splits,
            test_size=test_size,
            train_size=train_size+validate_size,
            random_state=random_state)

        if self.validate_set:
            total_size = validate_size + train_size
            self.validate_spliter = StratifiedShuffleSplit(
                n_splits=1, test_size=validate_size/total_size, train_size=train_size/total_size, random_state=random_state)

    def split(self, X, y, groups=None):
        r"""Generate indices to split data into training, validate and test set.

        Parameters
        ----------
        X : (n_samples, n_features) array_like
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.
        y : (n_samples,) array_like
            The target variable for supervised learning problems.
            Stratification is done based on the y labels.
        groups : optional
            Always ignored, exists for compatibility.

        Yields
        ------
        train : ndarray
            The training set indices for that split.
        validate : ndarray
            The validating set indices for that split.
            Available only if validate_set is True.
        test : ndarray
            The testing set indices for that split.
        """
        for train, test in super().split(X, y, groups=groups):
            if self.validate_set:
                train_idx, validate_idx = next(self.validate_spliter.split(X[train], y[train], groups=groups))
                yield train[train_idx], train[validate_idx], test
            else:
                yield train, test
