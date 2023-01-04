# -*- coding: utf-8 -*-
#
# Authors: Swolf <swolfforever@gmail.com>
# Date: 2022/12/01
# License: MIT License
"""Base Dataset Design.
"""
from abc import abstractmethod

class BaseDataset:
    r"""Base Dataset.

    Parameters
    ----------
    dataset_uid : str
        The unique string id to identify the dataset.
    subject_ids : list
        A list of available subject ids.
    valid_paradigms : list
        A list of valid paradigm uids.

    Attributes
    ----------
    uid : str
        The unique id for the current dataset.
    subjects : list
        All available subject ids.
    paradigms : list
        All valid paradigm uids.
    """
    def __init__(self,
        dataset_uid, subject_ids, valid_paradigms):
        self._dataset_uid = dataset_uid
        self._subject_ids = subject_ids
        self._valid_paradigms = valid_paradigms

    @property
    def subjects(self):
        return self._subject_ids

    @property
    def paradigms(self):
        return self._valid_paradigms

    @property
    def uid(self):
        return self._dataset_uid

    @abstractmethod
    def data_path(self,
        subject_id,
        local_path=None,
        force_update=False,
        proxies=None):
        r"""Mapping remote data to local and return the local file path.

        Parameters
        ----------
        subject_id : int
            The subject id.
        local_path : str, optional
            The local path to store remote data.
            If None, the default path is the psygine_data folder under the home directory.
        force_update : bool
            Whether to force update local files, default False.
        proxies : dict
            Proxy settings from the Request package.
        """

    @abstractmethod
    def _get_single_subject_data(self,
        subject_id):
        r"""Get a single subject's raw data.

        Parameters
        ----------
        subject_id : int
            The subject id.

        Returns
        -------
        dict
            A dictionary containing raw data, which are structured in session->run order.
        """

    def get_rawdata(self, subject_ids=None):
        r"""Get raw data from multiple subjects.

        Parameters
        ----------
        subject_ids : list, optional
            A list of selected subject ids. If None, use all available subjects in the dataset.

        Returns
        -------
        dict
            A dictionary containing raw data, which are structured in subject->session->run order.
        """
        if subject_ids is None:
            # use all subjects if not provided
            subject_ids = self.subjects
        else:
            # else check if the subject is valid
            for subject_id in subject_ids:
                if subject_id not in self.subjects:
                    raise ValueError('Invalid subject {}.'.format(subject_id))

        rawdata = dict()
        for subject_id in subject_ids:
            rawdata['subject_{:d}'.format(subject_id)] = self._get_single_subject_data(subject_id)
        return rawdata

    def download_all(self,
        local_path=None,
        force_update=False,
        proxies=None):
        r"""Download all subjects' data.

        parameters
        ----------
        local_path : str, optional
            The local path to store remote data.
            If None, the default path is the psygine_data folder under the home directory.
        force_update : bool
            Whether to force update local files, default False.
        proxies : dict
            Proxy settings from the Request package.
        """
        for subject_id in self.subjects:
            self.data_path(
                subject_id,
                local_path=local_path,
                force_update=force_update,
                proxies=proxies)
