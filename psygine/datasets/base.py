# -*- coding: utf-8 -*-
#
# Authors: Swolf <swolfforever@gmail.com>
# Date: 2023/10/02
# License: MIT License
"""Base Dataset Design.
"""
from abc import abstractmethod


class BaseDataset:
    r"""Base Dataset."""

    def __init__(self, dataset_uid):
        self.__dataset_uid = dataset_uid

    @property
    def uid(self):
        return self.__dataset_uid

    @abstractmethod
    def data_path(self, local_path=None, force_update=False, proxies=None):
        pass

    @abstractmethod
    def get_data(self):
        pass
