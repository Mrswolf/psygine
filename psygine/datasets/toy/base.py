# -*- coding: utf-8 -*-
#
# Authors: Swolf <swolfforever@gmail.com>
# Date: 2023/10/05
# License: MIT License
"""Base Toy Dataset Design.
"""
from abc import abstractmethod

from ..base import BaseDataset


class BaseToyVideoDataset(BaseDataset):
    def __init__(self, dataset_uid):
        super().__init__(dataset_uid)

    def download_all(self, local_path=None, force_update=False, proxies=None):
        self.data_path(
            local_path=local_path,
            force_update=force_update,
            proxies=proxies,
        )
