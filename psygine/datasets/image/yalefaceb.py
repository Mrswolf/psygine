# -*- coding: utf-8 -*-
#
# Authors: Swolf <swolfforever@gmail.com>
# Date: 2023/03/12
# License: MIT License
"""Base Yale Face Dataset Design.
"""
import os.path as op
from pathlib import Path
from collections import OrderedDict

from .base import ImageDataset
from ..utils.network import get_data_path
from ..utils.io import loadmat
from scipy.io import savemat

import numpy as np

YALEFACEB_URL = "https://github.com/Franjcf/Data-Science-Projects/raw/main/face_recognition_PCA/YALEBXF.mat"


class YaleFaceBDataset(ImageDataset):
    """YaleFaceBDataset.

    Total 38 subjects (no subject 14). Subjects 11, 12, 13, 15, 16, 17, and 18 have different numbers of images.
    """

    def __init__(self, local_path=None):
        super().__init__("yaleface_b", local_path=local_path)
        self.subjects = list(range(1, 14)) + list(range(15, 40))

    def __len__(self):
        return len(self.subjects)

    def _data_path(self, idx, local_path=None, force_update=False, proxies=None):
        if idx < 0 and idx >= len(self.subjects):
            raise ValueError("Index out of range")

        file_dest = get_data_path(
            YALEFACEB_URL,
            "yaleface",
            path=local_path,
            proxies=proxies,
            force_update=force_update,
        )
        file_path = op.join(
            Path(file_dest).parent, "person_{:02d}.mat".format(self.subjects[idx])
        )
        if not op.exists(file_path):
            raw = loadmat(file_dest)
            X, Y = raw["X"], raw["Y"]
            for i in self.subjects:
                images = X[:, Y == i]
                images = np.reshape(images, (192, 168, -1), order="F")
                images = np.transpose(images, (2, 0, 1))
                savemat(
                    op.join(Path(file_dest).parent, "person_{:02d}.mat".format(i)),
                    {"images": images},
                )
        return file_path

    def _load_data(self, path):
        images = loadmat(path)["images"]
        return images
