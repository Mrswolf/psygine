# -*- coding: utf-8 -*-
#
# Authors: Swolf <swolfforever@gmail.com>
# Date: 2022/12/31
# License: MIT License
"""Brain/Neuro Computer Interface (BNCI) datasets.
"""
import numpy as np
from mne import create_info
from mne.io import RawArray
from mne.channels import make_standard_montage
from .base import BaseEEGDataset
from ..utils.network import get_data_path
from ..utils.io import loadmat

BNCI_URL = "http://bnci-horizon-2020.eu/database/data-sets/"


class BNCI2014001(BaseEEGDataset):
    _EVENTS = {
        "left_hand": (1, (2, 6)), 
        "right_hand": (2, (2, 6)), 
        "feet": (3, (2, 6)), 
        "tongue": (4, (2, 6))
    }

    _CHANNELS = [
        'FZ', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 
        'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 
        'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 
        'P1', 'PZ', 'P2', 'POZ'
    ]
    def __init__(self, local_path=None):
        super().__init__(
            uid="bnci2014001",
            subjects=list(range(0, 9)),
            paradigms=["mi-eeg"],
            events=self._EVENTS,
            channels=self._CHANNELS,
            srate=250,
            local_path=local_path,
        )

    def _data_path(self,
        subject_id,
        local_path=None,
        force_update=False,
        proxies=None):
        if subject_id not in self.subjects:
            raise IndexError("Invalid subject id")

        url = '{:s}001-2014/A{:02d}'.format(BNCI_URL, subject_id+1)

        dests = [
            [
                get_data_path('{:s}{:s}.mat'.format(url, 'E'), self.uid, 
                path=local_path, proxies=proxies, force_update=force_update)
            ],
            [
                get_data_path('{:s}{:s}.mat'.format(url, 'T'), self.uid, 
                path=local_path, proxies=proxies, force_update=force_update)
            ]
        ]

        return dests

    def _get_single_subject_data(self, subject_id):
        dests = self._data_path(subject_id, local_path=self.local_path)
        montage = make_standard_montage('standard_1005')
        montage.rename_channels({ch_name: ch_name.upper() for ch_name in montage.ch_names})

        sess = dict()
        for isess, run_dests in enumerate(dests):
            run_arrays = loadmat(run_dests[0])['data']
            runs = dict()
            for irun, run_array in enumerate(run_arrays):
                X = run_array['X'].T * 1e-6 # volt
                trial = run_array['trial']
                y = run_array['y']
                stim =  np.zeros((1, X.shape[-1]))

                if y.size > 0:
                    stim[0, trial-1] = y

                data = np.concatenate((X, stim), axis=0)

                ch_names = [ch_name.upper() for ch_name in self.channels] + ['EOG1', 'EOG2', 'EOG3']
                ch_types = ['eeg']*len(self.channels) + ['eog']*3
                ch_names = ch_names + ['STI 014']
                ch_types = ch_types + ['stim']

                info = create_info(
                    ch_names, self.srate,
                    ch_types=ch_types,
                    verbose=False
                    )
                raw = RawArray(data, info, verbose=False)
                # raw = raw.rename_channels({ch_name: ch_name.upper() for ch_name in raw.info['ch_names']})
                raw.set_montage(montage, verbose=False)
                runs['run_{:d}'.format(irun)] = raw
            sess['session_{:d}'.format(isess)] = runs
        return sess


class BNCI2014004(BaseEEGDataset):
    _EVENTS = {
        "left_hand": (1, (3, 7.5)), 
        "right_hand": (2, (3, 7.5)), 
    }

    _CHANNELS = [
        'C3', 'CZ', 'C4'
    ]
    def __init__(self, local_path=None):
        super().__init__(
            uid="bnci2014004",
            subjects=list(range(0, 9)),
            paradigms=["mi-eeg"],
            events=self._EVENTS,
            channels=self._CHANNELS,
            srate=250,
            local_path=local_path,
        )

    def _data_path(self,
        subject_id,
        local_path=None,
        force_update=False,
        proxies=None):
        if subject_id not in self.subjects:
            raise IndexError("Invalid subject id")

        url = '{:s}004-2014/B{:02d}'.format(BNCI_URL, subject_id+1)

        dests = [
            [
                get_data_path('{:s}{:s}.mat'.format(url, 'E'), self.uid, 
                path=local_path, proxies=proxies, force_update=force_update)
            ],
            [
                get_data_path('{:s}{:s}.mat'.format(url, 'T'), self.uid, 
                path=local_path, proxies=proxies, force_update=force_update)
            ]
        ]

        return dests

    def _get_single_subject_data(self, subject_id):
        dests = self._data_path(subject_id, self.local_path)
        montage = make_standard_montage('standard_1005')
        montage.rename_channels({ch_name: ch_name.upper() for ch_name in montage.ch_names})

        sess_arrays = loadmat(dests[0][0])['data'] + loadmat(dests[1][0])['data']

        sess = dict()
        for isess, sess_array in enumerate(sess_arrays):
            runs = dict()
            X = sess_array['X'].T * 1e-6 # volt
            trial = sess_array['trial']
            y = sess_array['y']
            stim = np.zeros((1, X.shape[-1]))

            if y.size > 0:
                stim[0, trial-1] = y

            data = np.concatenate((X, stim), axis=0)

            ch_names = [ch_name.upper() for ch_name in self.channels] + ['EOG1', 'EOG2', 'EOG3']
            ch_types = ['eeg']*len(self.channels) + ['eog']*3
            ch_names = ch_names + ['STI 014']
            ch_types = ch_types + ['stim']

            info = create_info(
                ch_names, self.srate,
                ch_types=ch_types,
                verbose=False
                )
            raw = RawArray(data, info, verbose=False)
            # raw = raw.rename_channels({ch_name: ch_name.upper() for ch_name in raw.info['ch_names']}, verbose=False)
            raw.set_montage(montage, verbose=False)
            runs['run_0'] = raw
            sess['session_{:d}'.format(isess)] = runs
        return sess
