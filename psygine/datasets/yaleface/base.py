# -*- coding: utf-8 -*-
#
# Authors: Swolf <swolfforever@gmail.com>
# Date: 2023/03/12
# License: MIT License
"""Base Yale Face Dataset Design.
"""
from abc import abstractmethod

from ..base import BaseDataset


class BaseYaleFaceDataset(BaseDataset):
    def __init__(self, dataset_uid, subjects):
        super().__init__(dataset_uid)
        self.__subject_ids = subjects

    @property
    def subjects(self):
        return self.__subject_ids

    @abstractmethod
    def data_path(self, subject_id, local_path=None, force_update=False, proxies=None):
        pass

    @abstractmethod
    def get_data(self, subject_ids=None):
        pass

    def download_all(self, local_path=None, force_update=False, proxies=None):
        for subject_id in self.subjects:
            self.data_path(
                subject_id,
                local_path=local_path,
                force_update=force_update,
                proxies=proxies,
            )
