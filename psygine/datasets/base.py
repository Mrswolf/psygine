# -*- coding: utf-8 -*-
#
# Authors: Swolf <swolfforever@gmail.com>
# Date: 2023/10/02
# License: MIT License
"""Base Dataset Design.
"""
from abc import abstractmethod, ABC


class BaseDataset(ABC):
    r"""Base Dataset."""

    def __init__(self, dataset_uid):
        self.__dataset_uid = dataset_uid

    @property
    def uid(self):
        return self.__dataset_uid

    @abstractmethod
    def __getitem__(self, idx):
        pass

    @abstractmethod
    def __len__(self):
        pass
