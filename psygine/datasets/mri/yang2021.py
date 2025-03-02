# -*- coding: utf-8 -*-
#
# Authors: Swolf <swolfforever@gmail.com>
# Date: 2025/02/26
# License: MIT License
"""ADMM-CSNet Dataset.
"""
import os
import numpy as np
import gdown
import rarfile
from ..base import BaseDataset
from ..utils.io import loadmat, get_local_path, set_local_path

YANG2021_URL = "https://drive.google.com/drive/folders/1UhQ01pdmO11Agc5sM61Mt7KQTN9LytNt?usp=sharing"


class _BaseYang2021(BaseDataset):
    def __init__(self, isTrain=False, isValid=False, isTest=False, local_path=None):
        super().__init__("yang2021")
        self.isTrain = isTrain
        self.isValid = isValid
        self.isTest = isTest
        self.local_path = local_path

    def __len__(self):
        if self.isTrain:
            return 99
        if self.isValid:
            return 50
        if self.isTest:
            return 50

    def __getitem__(self, idx):
        data = self._load_data(
            self._data_path(idx, local_path=self.local_path, force_update=False)
        )
        return data["data"]["data"]["train"], data["data"]["data"]["label"]

    def _data_path(self, idx, local_path=None, force_update=False, proxies=None):
        key = "PSYGINE_DATASETS_{:s}_PATH".format(self.uid.upper())
        key_dest = "psygine_{:s}_data".format(self.uid.lower())

        if local_path is None:
            local_path = get_local_path(key)

        if self.isTrain:
            dest_folder = os.path.join(local_path, key_dest, "train")
        if self.isValid:
            dest_folder = os.path.join(local_path, key_dest, "valid")
        if self.isTest:
            dest_folder = os.path.join(local_path, key_dest, "test")

        if not os.path.exists(dest_folder + ".rar") or force_update:
            gdown.download_folder(
                YANG2021_URL,
                output=local_path,
                quiet=True,
                resume=False if force_update else True,
            )
        if not os.path.exists(dest_folder):
            with rarfile.RarFile(dest_folder + ".rar") as rf:
                rf.extractall(path=os.path.join(local_path, key_dest))
        set_local_path(key, local_path)

        dest = os.path.join(dest_folder, f"new{idx+1:02d}.mat")
        return dest

    def _load_data(self, path):
        data = loadmat(path)
        return data


class Yang2021_Train(_BaseYang2021):
    def __init__(self, local_path=None):
        super().__init__(isTrain=True, local_path=local_path)


class Yang2021_Valid(_BaseYang2021):
    def __init__(self, local_path=None):
        super().__init__(isValid=True, local_path=local_path)


class Yang2021_Test(_BaseYang2021):
    def __init__(self, local_path=None):
        super().__init__(isTest=True, local_path=local_path)
