# -*- coding: utf-8 -*-
#
# Authors: Swolf <swolfforever@gmail.com>
# Date: 2022/12/01
# License: MIT License
"""Base Paradigm Design.
"""
from abc import abstractmethod
from joblib import Parallel, delayed


class BaseParadigm:
    r"""Base Paradigm.

    Parameters
    ----------
    uid : str
        The unique string id to identify the paradigm.

    Attributes
    ----------
    uid : str
        The unique id for the current paradigm.
    """

    def __init__(self, uid):
        self._paradigm_uid = uid

    @abstractmethod
    def is_valid(self, dataset):
        r"""check the validity of the dataset.

        Parameters
        ----------
        dataset : BaseDataset
            An instance of BaseDataset.

        Returns
        -------
        bool
            Return True if the dataset is valid, else False.
        """

    @abstractmethod
    def _get_single_subject_data(self, dataset, subject_id):
        r"""Get data from a single subject.

        Parameters
        ----------
        dataset : BaseDataset
            An instance of BaseDataset.
        subject_id : int
            The subject's id.

        Returns
        -------
        X : Any
            Data, usually a numpy array object.
        y : Any
            Label, usually a numpy array object.
        meta : Any
            Nearly a database storing useful information, usually a pandas dataframe object.
        """

    def get_data(self, dataset, subject_ids=None, n_jobs=None):
        r"""Get data from multiple subjects.

        Parameters
        ----------
        dataset : BaseDataset
            An instance of BaseDataset.
        subject_ids : list, optional
            A list of queried subject ids. If None, use all subjects the dataset supported.
        n_jobs : int, optional
            The number of cores to use to load data, -1 for all cores.

        Returns
        -------
        X_list : array_like
            A list of X.
        y_list : array_like
            A list of y.
        meta_list : DataFrame
            A list of meta.
        """
        if not self.is_valid(dataset):
            raise TypeError(
                "Dataset {:s} is not valid for the current paradigm.\nCheck paradigm settings.".format(
                    dataset.uid
                )
            )
        if subject_ids is None:
            subject_ids = dataset.subjects
        X_list, y_list, meta_list = zip(
            *Parallel(n_jobs=n_jobs)(
                delayed(self._get_single_subject_data)(dataset, subject_id)
                for subject_id in subject_ids
            )
        )
        return X_list, y_list, meta_list

    @property
    def uid(self):
        return self._paradigm_uid
