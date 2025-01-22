# -*- coding: utf-8 -*-
#
# Authors: Swolf <swolfforever@gmail.com>
# Date: 2023/10/05
# License: MIT License
"""Traffic Video Datasets.

"""
from collections import OrderedDict

from ..base import BaseDataset
from ..utils.network import get_data_path

import numpy as np
import cv2


TRAFFIC1_URL = "https://github.com/andrewssobral/lrslibrary/raw/master/dataset/demo.avi"


class TrafficVideo1Dataset(BaseDataset):
    def __init__(self, local_path=None):
        super().__init__("traffic1")
        self.local_path = local_path

    def __len__(self):
        return 1

    def _data_path(self, local_path=None, force_update=False, proxies=None):
        file_dest = get_data_path(
            TRAFFIC1_URL,
            "misc",
            path=local_path,
            proxies=proxies,
            force_update=force_update,
        )
        return file_dest

    def __getitem__(self, idx):
        path = self._data_path(local_path=self.local_path)
        video = cv2.VideoCapture(path)
        if not video.isOpened():
            raise Exception("Failed to open video file.")
        frames = []
        while True:
            retval, frame = video.read()
            if not retval:
                break
            frames.append(frame)
        video.release()

        frames = np.array(frames)
        return frames
