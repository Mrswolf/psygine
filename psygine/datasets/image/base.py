from abc import abstractmethod
import numpy as np
from ..base import BaseDataset


class ImageDataset(BaseDataset):
    def __init__(self, dataset_uid, local_path=None):
        super().__init__(dataset_uid)
        self.local_path = local_path

    def __getitem__(self, idx):
        image = self._load_data(self._data_path(idx, local_path=self.local_path))
        return image

    @abstractmethod
    def _load_data(self, path):
        pass

    @abstractmethod
    def _data_path(self, idx, local_path=None, force_update=False, proxies=None):
        pass


class PairImageDataset(BaseDataset):
    def __init__(self, dataset_uid, local_path=None):
        super().__init__(dataset_uid)
        self.local_path = local_path

    def __getitem__(self, idx):
        imageX = self._load_data(self._data_path(idx, local_path=self.local_path)[0])
        imageY = self._load_data(self._data_path(idx, local_path=self.local_path)[1])
        return imageX, imageY
    
    @abstractmethod
    def _load_data(self, path):
        pass

    @abstractmethod
    def _data_path(self, idx, local_path=None, force_update=False, proxies=None):
        pass
