# -*- coding: utf-8 -*-
#
# Authors: Swolf <swolfforever@gmail.com>
# Date: 2023/10/02
# License: MIT License
"""Base Text Dataset Design.
"""
from abc import abstractmethod

from ..base import BaseDataset


class BaseMovieLensDataset(BaseDataset):
    def __init__(self, dataset_uid, tables):
        super().__init__(dataset_uid)
        self.__tables = tables  # a list of available tables

    @property
    def tables(self):
        return self.__tables

    @abstractmethod
    def get_data(self, tables=None):
        pass

    def download_all(self, local_path=None, force_update=False, proxies=None):
        self.data_path(
            local_path=local_path,
            force_update=force_update,
            proxies=proxies,
        )
