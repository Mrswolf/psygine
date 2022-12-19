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
    def __init__(self, uid):
        self._paradigm_uid = uid

    @abstractmethod
    def is_valid(self, dataset):
        pass
    
    @abstractmethod
    def _get_single_subject_data(self, dataset, subject_id):
        pass

    def get_data(self, dataset, subject_ids=None, n_jobs=None):
        if not self.is_valid(dataset):
            raise TypeError(
                "Dataset {:s} is not valid for the current paradigm.\nCheck paradigm settings.".format(dataset.uid))
        if subject_ids is None:
            subject_ids = dataset.subjects
        X_list, y_list, meta_list = zip(
            *Parallel(n_jobs=n_jobs)(
                delayed(self._get_single_subject_data)(dataset, subject_id) for subject_id in subject_ids))
        return X_list, y_list, meta_list
    
    @property
    def uid(self):
        return self._paradigm_uid