# -*- coding: utf-8 -*-
#
# Authors: Swolf <swolfforever@gmail.com>
# Date: 2023/03/12
# License: MIT License
"""Yale Face B Dataset.

The original link is dead now. Here are my modifed version from 
https://github.com/Franjcf/Data-Science-Projects/blob/main/face_recognition_PCA/YALEBXF.mat. The http download is not working and i don't know why.
"""
import os.path as op
from collections import OrderedDict

from .base import BaseYaleFaceDataset
from ..utils.network import get_data_path
from ..utils.io import loadmat

import numpy as np

# YALEFACEB_URL = "http://cornelltech.github.io/cs5785-fall-2017/data/faces.zip"
# YALEFACEB_URL = "https://github.com/Franjcf/Data-Science-Projects/blob/main/face_recognition_PCA/YALEBXF.mat"


class YaleFaceBDataset(BaseYaleFaceDataset):
    """YaleFaceBDataset.

    Total 38 subjects (no subject 14). Subjects 11, 12, 13, 15, 16, 17, and 18 have different numbers of images.
    """

    def __init__(self):
        super().__init__("yaleface_b", list(range(1, 14)) + list(range(15, 40)))

    def data_path(self, subject_id, local_path=None, force_update=False, proxies=None):
        url = op.join(
            op.dirname(__file__), "data", "person_{:02d}.mat".format(subject_id)
        )
        file_dest = get_data_path(
            url,
            "yaleface",
            path=local_path,
            proxies=proxies,
            force_update=force_update,
        )
        return [file_dest]

    def get_data(self, subject_ids=None):
        if subject_ids is None:
            subject_ids = self.subjects

        rawdata = OrderedDict()
        for subject_id in subject_ids:
            files = self.data_path(subject_id)
            images = loadmat(files[0])["images"]
            rawdata[subject_id] = images

        return rawdata
