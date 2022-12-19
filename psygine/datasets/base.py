# -*- coding: utf-8 -*-
#
# Authors: Swolf <swolfforever@gmail.com>
# Date: 2022/12/01
# License: MIT License
"""Base Dataset Design.
"""
from abc import abstractmethod

class BaseDataset:
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
        r"""Mapping remote data to local and return local path.
        """

    @abstractmethod
    def _get_single_subject_data(self,
        subject_id):
        r"""return raw data structured in subject->session->run
        """
    
    def get_rawdata(self, subject_ids=None):
        r"""Get raw data in subject_ids. If None return all available subjects' data.
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
        r"""Download all data.
        """
        for subject_id in self.subjects:
            self.data_path(
                subject_id,
                local_path=local_path,
                force_update=force_update,
                proxies=proxies)