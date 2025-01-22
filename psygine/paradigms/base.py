# -*- coding: utf-8 -*-
#
# Authors: Swolf <swolfforever@gmail.com>
# Date: 2022/12/01
# License: MIT License
"""Base Paradigm Design.
"""
from abc import abstractmethod, ABC
from joblib import Parallel, delayed


class BaseParadigm(ABC):
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
    def _process_data(self, dataset, idx):
        r"""Get processed data from index.

        Parameters
        ----------
        dataset : BaseDataset
            An instance of BaseDataset.
        idx : int
            Any index to query data.

        Returns
        -------
        X : Any
            Data, usually a numpy array object.
        y : Any
            Label, usually a numpy array object.
        meta : Any
            Nearly a database storing useful information, usually a pandas dataframe object.
        """

    def get_data(self, dataset, idxs=None, n_jobs=None):
        r"""Get data from multiple subjects.

        Parameters
        ----------
        dataset : BaseDataset
            An instance of BaseDataset.
        idxs : list, optional
            A list of queried indexes. If None, use all indexes the dataset supported.
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
        if idxs is None:
            idxs = list(range(len(dataset)))
        # X_list, y_list, meta_list = zip(
        #     *Parallel(n_jobs=n_jobs)(
        #         delayed(self._process_data)(dataset, idx) for idx in idxs
        #     )
        # )

        outputs = Parallel(n_jobs=n_jobs)(
            delayed(self._process_data)(dataset, idx) for idx in idxs
        )
        return outputs

    @property
    def uid(self):
        return self._paradigm_uid
