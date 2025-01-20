# -*- coding: utf-8 -*-
#
# Authors: Swolf <swolfforever@gmail.com>
# Date: 2023/10/02
# License: MIT License
"""Base MovieLens Dataset Design.
"""
from abc import abstractmethod

import pandas as pd

from ..base import BaseDataset

class BaseMovieLensDataset(BaseDataset):

    def __init__(self, dataset_uid, table_cols, local_path=None):
        super().__init__(dataset_uid)
        self.local_path = local_path
        self._table_cols = table_cols
        dests = self._data_path(local_path=self.local_path)
        pd_tables = self._get_rawdata(dests)
        self._pd_tables = pd.concat(
            [pd_tables[table] for table in self.get_tables()], axis=1
        )

    @abstractmethod
    def _data_path(self, local_path=None, force_update=False, proxies=None):
        pass

    @abstractmethod
    def _get_rawdata(self, dests):
        pass

    def get_columns(self, table):
        return self._table_cols[table]

    def get_tables(self):
        return self._table_cols.keys()

    def get_rawdata(self):
        return self._pd_tables

    def __getitem__(self, idx):
        return self._pd_tables.iloc[idx]
