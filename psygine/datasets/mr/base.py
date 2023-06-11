# -*- coding: utf-8 -*-
#
# Authors: Swolf <swolfforever@gmail.com>
# Date: 2023/03/12
# License: MIT License
"""Base MR Dataset Design.
"""
from ..base import BaseDataset

# ocmr hint: https://ocmr.s3.amazonaws.com/data/fs_0001_1_5T.h5
# https://ocmr.s3.amazonaws.com/ocmr_data_attributes.csv

class MriDataset(BaseDataset):
    r"""Base MRI Dataset.



    """
    def __init__(self,
        uid,
        subjects,
        paradigms):
        super().__init__(uid, subjects, paradigms)
    
    def data_path(
            self,
            subject_id,
            local_path=None,
            force_update=False,
            proxies=None):