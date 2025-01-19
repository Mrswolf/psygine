# -*- coding: utf-8 -*-
#
# Authors: Swolf <swolfforever@gmail.com>
# Date: 2024/12/27
# License: MIT License
"""GoPro Deblurring Dataset Design.
"""
import os
import os.path as op
from pathlib import Path
from joblib import Parallel, delayed
import numpy as np
import zipfile
from PIL import Image
from .base import PairImageDataset
from ..utils.network import get_data_path

GOPRO_GOOGLE_URL = "https://drive.google.com/uc?export=download&id=1y4wvPdOG3mojpFCHTqLgriexhbjoWVkK"
GOPRO_SNU_URL = "http://data.cv.snu.ac.kr:8008/webdav/dataset/GOPRO/GOPRO_Large.zip"
GOPRO_LOCAL_URL = "/GOPRO_Large.zip"


class _GoProDataset(PairImageDataset):
    def __init__(self, isTrain=True, local_path=None, prefetch=False):
        super().__init__("gopro", local_path=local_path)
        self.isTrain = isTrain
        self.imageX_list = None
        self.imageY_list = None
        self.prefetch = prefetch

        if self.prefetch:
            self._data = Parallel(n_jobs=-1)(
                delayed(
                    lambda idx: super(_GoProDataset, self).__getitem__(idx))(idx) 
                    for idx in range(len(self)))

    def __getitem__(self, idx):
        if self.prefetch:
            return self._data[idx]
        else:
            return super(_GoProDataset, self).__getitem__(idx)

    def __len__(self):
        if self.isTrain:
            return 2103
        else:
            return 1111

    def _data_path(self, idx, local_path=None, force_update=False, proxies=None):
        if self.imageY_list is not None and self.imageX_list is not None:
            return (self.imageX_list[idx], self.imageY_list[idx])
        
        if local_path is not None:
            self.local_path = local_path
        file_dest = get_data_path(GOPRO_LOCAL_URL, "gopro", self.local_path, force_update, proxies)
        parent_dir = Path(file_dest).parent
        domain = "train" if self.isTrain else "test"
        if not op.exists(op.join(parent_dir, domain)):
            with zipfile.ZipFile(file_dest, "r") as archive:
                archive.extractall(path=parent_dir)

        X, Y = [], []
        for root, dirs, _ in os.walk(op.join(parent_dir, domain)):

            if 'sharp' in dirs and 'blur' in dirs:
                sharp_folder = os.path.join(root, 'sharp')
                blur_folder = os.path.join(root, 'blur')

                # get all images files with png extension from sharp folder
                imagefiles = list(zip(
                    *[
                        (os.path.join(blur_folder, f), os.path.join(sharp_folder, f)) 
                        for f in sorted(os.listdir(sharp_folder)) if f.endswith('.png')]))
                X.extend(imagefiles[0])
                Y.extend(imagefiles[1])
        self.imageX_list = X
        self.imageY_list = Y
        return (self.imageX_list[idx], self.imageY_list[idx])
    
    def _load_data(self, path):
        return np.array(Image.open(path))
    

class GoProTrainDataset(_GoProDataset):
    def __init__(self, local_path=None, prefetch=False):
        super().__init__(isTrain=True, local_path=local_path, prefetch=prefetch)

class GoProTestDataset(_GoProDataset):
    def __init__(self, local_path=None, prefetch=False):
        super().__init__(isTrain=False, local_path=local_path, prefetch=prefetch)


