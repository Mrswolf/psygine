# -*- coding: utf-8 -*-
#
# Authors: Swolf <swolfforever@gmail.com>
# Date: 2022/12/13
# License: MIT License
"""Nakanishi SSVEP dataset.
"""
from mne.io import Raw
from mne.channels import make_standard_montage
from .base import BaseEEGDataset
from ..utils.network import get_data_path

ALEX_URL = "https://zenodo.org/record/806023/files/"


class AlexMI(BaseEEGDataset):
    _EVENTS = {
        "right_hand": (2, (0, 3)),
        "feet": (3, (0, 3)),
        "rest": (4, (0, 3))
    }

    _CHANNELS = [
        'FPZ','F7','F3','FZ','F4','F8',
        'T7','C3','C4','T8',
        'P7','P3','PZ','P4','P8'
    ]
    def __init__(self, local_path=None):
        super().__init__(
            uid="alexeeg",
            subjects=list(range(0, 8)),
            paradigms=["mi-eeg"],
            events=self._EVENTS,
            channels=self._CHANNELS,
            srate=512,
            local_path=local_path,
        )

    def _data_path(self,
        subject_id,
        local_path=None,
        force_update=False,
        proxies=None):
        if subject_id not in self.subjects:
            raise IndexError("Invalid subject id")

        url = '{:s}subject{:d}.raw.fif'.format(ALEX_URL, subject_id+1)
        dests = [
            [get_data_path(url, self.uid,
                path=local_path, proxies=proxies, force_update=force_update)]
            ]
        return dests

    def _get_single_subject_data(self, subject_id):
        dests = self._data_path(subject_id, local_path=self.local_path)
        montage = make_standard_montage('standard_1005')
        montage.rename_channels({ch_name: ch_name.upper() for ch_name in montage.ch_names})

        sess = dict()
        for isess, run_dests in enumerate(dests):
            runs = dict()
            for irun, run_file in enumerate(run_dests):
                raw = Raw(run_file, preload=True, verbose=False)
                raw = raw.rename_channels({ch_name: ch_name.upper() for ch_name in raw.info['ch_names']}, verbose=False)
                raw.set_montage(montage, verbose=False)
                runs['run_{:d}'.format(irun)] = raw
            sess['session_{:d}'.format(isess)] = runs
        return sess
