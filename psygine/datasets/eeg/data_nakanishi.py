# -*- coding: utf-8 -*-
#
# Authors: Swolf <swolfforever@gmail.com>
# Date: 2022/12/12
# License: MIT License
"""Nakanishi SSVEP dataset.
"""
import numpy as np
from mne import create_info
from mne.io import RawArray
from mne.channels import make_standard_montage
from .base import BaseEEGDataset, SsvepMixin
from ..utils.network import get_data_path
from ..utils.io import loadmat

Nakanishi2015_URL = "https://github.com/mnakanishi/12JFPM_SSVEP/raw/master/data/"


class Nakanishi2015(SsvepMixin, BaseEEGDataset):
    _CHANNELS = ['PO7', 'PO3', 'POZ', 'PO4', 'PO8', 'O1', 'OZ', 'O2']
    _FREQS = [
        9.25, 11.25, 13.25, 9.75, 11.75, 13.75, 10.25, 12.25, 14.25, 10.75, 12.75, 14.75
    ]
    _PHASES = [
        0, 0, 0, 0.5, 0.5, 0.5, 1, 1, 1, 1.5, 1.5, 1.5
    ]
    _FREQ_PHASE_TABLE = {
        "{:.1f}/{:.1f}".format(param[0], param[1]): (param[0],  param[1]) for param in zip(_FREQS, _PHASES)
    }
    _EVENTS = {
        "{:.1f}/{:.1f}".format(param[0], param[1]): (i+1, (0, 4)) for i, param in enumerate(zip(_FREQS, _PHASES))
    }
    def __init__(self, local_path=None):
        super().__init__(
            uid="nakanishi2015",
            subjects=list(range(0, 10)),
            paradigms=["ssvep-eeg"],
            events=self._EVENTS,
            channels=self._CHANNELS,
            srate=256,
            freq_phase_table=self._FREQ_PHASE_TABLE,
            local_path=local_path,
        )

    def _data_path(self,
        subject_id,
        local_path=None,
        force_update=False,
        proxies=None):
        if subject_id not in self.subjects:
            raise IndexError("Invalid subject id.")

        url = '{:s}s{:d}.mat'.format(Nakanishi2015_URL, subject_id+1)
        file_dest = get_data_path(url, self.uid, 
            path=local_path, proxies=proxies, force_update=force_update)

        dests = [
            [
                file_dest
            ]
        ]
        return dests

    def _get_single_subject_data(self, subject_id):
        montage = make_standard_montage('standard_1005')
        montage.rename_channels({ch_name: ch_name.upper() for ch_name in montage.ch_names})
        dests = self._data_path(subject_id, self.local_path)
        raw_mat = loadmat(dests[0][0])
        n_samples, n_channels, n_trials = 1114, 8, 15
        n_classes = 12

        data = np.transpose(raw_mat['eeg'], axes=(0, 3, 1, 2))
        data = np.reshape(data, newshape=(-1, n_channels, n_samples))
        data = data - data.mean(axis=2, keepdims=True)
        raw_events = np.zeros((data.shape[0], 1, n_samples))
        raw_events[:, 0, 38] = np.array([n_trials * [i + 1]
                                        for i in range(n_classes)]).flatten()
        data = np.concatenate([1e-6 * data, raw_events], axis=1)

        buff = (data.shape[0], n_channels + 1, 50)
        data = np.concatenate([np.zeros(buff), data,
                               np.zeros(buff)], axis=2)
        ch_names = self.channels + ['stim']
        ch_types = ['eeg']*len(self.channels) + ['stim']

        info = create_info(
            ch_names=ch_names, ch_types=ch_types, sfreq=self.srate, verbose=False
        )
        raw = RawArray(data=np.concatenate(list(data), axis=1), info=info, verbose=False)
        raw = raw.rename_channels({ch_name: ch_name.upper() for ch_name in raw.info['ch_names']}, verbose=False)
        raw.set_montage(montage, verbose=False)

        sess = {
            'session_0': {'run_0': raw}
        }
        return sess
