# -*- coding: utf-8 -*-
#
# Authors: Swolf <swolfforever@gmail.com>
# Date: 2023/03/12
# License: MIT License
"""Yale Face B Dataset.

https://github.com/Franjcf/Data-Science-Projects/blob/main/face_recognition_PCA/YALEBXF.mat.
"""
import os.path as op
from pathlib import Path
from collections import OrderedDict

from .base import BaseYaleFaceDataset
from ..utils.network import get_data_path
from ..utils.io import loadmat

import numpy as np
from scipy.io import savemat

YALEFACEB_URL = "https://github.com/Franjcf/Data-Science-Projects/raw/main/face_recognition_PCA/YALEBXF.mat"


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
            YALEFACEB_URL,
            "yaleface",
            path=local_path,
            proxies=proxies,
            force_update=force_update,
        )
        file_path = op.join(
            Path(file_dest).parent, "person_{:02d}.mat".format(subject_id)
        )
        if not op.exists(file_path):
            raw = loadmat(file_dest)
            X, Y = raw["X"], raw["Y"]
            for i in self.subjects:
                images = X[:, Y == i]
                images = np.reshape(images, (192, 168, -1), order="F")
                savemat(
                    op.join(Path(file_dest).parent, "person_{:02d}.mat".format(i)),
                    {"images": images},
                )
        return [file_path]

    def get_data(self, subject_ids=None):
        if subject_ids is None:
            subject_ids = self.subjects

        rawdata = OrderedDict()
        for subject_id in subject_ids:
            files = self.data_path(subject_id)
            images = loadmat(files[0])["images"]
            rawdata[subject_id] = images

        return rawdata
