# -*- coding: utf-8 -*-
#
# Authors: Swolf <swolfforever@gmail.com>
# Date: 2025/02/22
# License: MIT License
"""MOABB datasets.

"""
from .base import BaseEEGDataset

import importlib, sys
from pathlib import Path
from mne import get_config, set_config
from mne.datasets.utils import _do_path_update, _get_path


def get_dataset_path(sign, path):
    """Returns the dataset path allowing for changes in MNE_DATA config.

    Parameters
    ----------
    sign : str
        Signifier of dataset
    path : None | str
        Location of where to look for the data storing location.
        If None, the environment variable or config parameter
        ``MNE_DATASETS_(signifier)_PATH`` is used. If it doesn't exist, the
        "~/mne_data" directory is used. If the dataset
        is not found under the given path, the data
        will be automatically downloaded to the specified folder.

    Returns
    -------
        path : None | str
        Location of where to look for the data storing location
    """
    sign = sign.upper()
    key = "MNE_DATASETS_{:s}_PATH".format(sign)
    if get_config(key) is None:
        if get_config("MNE_DATA") is None:
            path_def = Path.home() / "mne_data"
            print(
                "MNE_DATA is not already configured. It will be set to "
                "default location in the home directory - "
                + str(path_def)
                + "\nAll datasets will be downloaded to this location, if anything is "
                "already downloaded, please move manually to this location"
            )
            if not path_def.is_dir():
                path_def.mkdir(parents=True)
            set_config("MNE_DATA", str(Path.home() / "mne_data"))
        set_config(key, get_config("MNE_DATA"))

    if path is None:
        path = get_config(key)

    _do_path_update(path, True, key, sign)  # force update path
    return _get_path(path, key, sign)


import moabb.datasets.download as dl

dl.get_dataset_path = get_dataset_path

imported_modules = [
    key
    for key in sys.modules.keys()
    if key.startswith("moabb.datasets") and sys.modules[key] is not dl
]
for key in imported_modules:
    importlib.reload(sys.modules[key])

from moabb.datasets import (
    BNCI2014_002,
    BNCI2015_001,
    BNCI2015_004,
    Cho2017,
    GrosseWentrup2009,
    Lee2019_MI,
    Liu2024,
    Ofner2017,
    # PhysionetMI,# broken apis
    Schirrmeister2017,
    Shin2017A,
    Shin2017B,
    Stieger2021,
    Weibo2014,
    Zhou2016,
)


def check_moabb_subjects(subjects):
    return [subject_id - 1 for subject_id in subjects]


__MOABB_PARADIGMS = {
    "p300": ["p300-eeg"],
    "imagery": ["mi-eeg"],
    "ssvep": ["ssvep-eeg"],
}


def check_moabb_paradigms(paradigm):
    return __MOABB_PARADIGMS[paradigm]


def check_moabb_events(events, interval):
    return {event: (eid, interval) for event, eid in events.items()}


def check_moabb_rawdata(rawdata):
    data = dict()
    for sess_id, (old_sess_id, runs) in enumerate(rawdata.items()):
        data[f"session_{sess_id}"] = dict()
        for run_id, (old_run_id, raw) in enumerate(runs.items()):
            data[f"session_{sess_id}"][f"run_{run_id}"] = raw
    return data


def moabb_wrapper(cls, srate=None, channels=None):
    class BaseMOABBDataset(BaseEEGDataset):

        def __init__(self, local_path=None, **kwargs):
            self._moabb = cls(**kwargs)
            super().__init__(
                self._moabb.code,
                check_moabb_subjects(self._moabb.subject_list),
                check_moabb_paradigms(self._moabb.paradigm),
                check_moabb_events(self._moabb.event_id, self._moabb.interval),
                channels,
                srate,
                local_path=local_path,
            )

        def _data_path(
            self, subject_id, local_path=None, force_update=False, proxies=None
        ):
            dests = self._moabb.data_path(
                subject_id + 1,
                path=local_path if self.local_path is None else self.local_path,
                force_update=force_update,
                update_path=True,
            )
            return dests

        def _get_single_subject_data(self, subject_id):
            _ = self._data_path(subject_id)  # load once to update store location
            data = self._moabb.get_data(subjects=[subject_id + 1])[subject_id + 1]
            data = check_moabb_rawdata(data)
            return data

    return BaseMOABBDataset


# not available now
Cho2017 = moabb_wrapper(
    Cho2017,
    512,
    [
        "FP1",
        "AF7",
        "AF3",
        "F1",
        "F3",
        "F5",
        "F7",
        "FT7",
        "FC5",
        "FC3",
        "FC1",
        "C1",
        "C3",
        "C5",
        "T7",
        "TP7",
        "CP5",
        "CP3",
        "CP1",
        "P1",
        "P3",
        "P5",
        "P7",
        "P9",
        "PO7",
        "PO3",
        "O1",
        "IZ",
        "OZ",
        "POZ",
        "PZ",
        "CPZ",
        "FPZ",
        "FP2",
        "AF8",
        "AF4",
        "AFZ",
        "FZ",
        "F2",
        "F4",
        "F6",
        "F8",
        "FT8",
        "FC6",
        "FC4",
        "FC2",
        "FCZ",
        "CZ",
        "C2",
        "C4",
        "C6",
        "T8",
        "TP8",
        "CP6",
        "CP4",
        "CP2",
        "P2",
        "P4",
        "P6",
        "P8",
        "P10",
        "PO8",
        "PO4",
        "O2",
    ],
)

Schirrmeister2017 = moabb_wrapper(
    Schirrmeister2017,
    500,
    [
        "FP1",
        "FP2",
        "FPZ",
        "F7",
        "F3",
        "FZ",
        "F4",
        "F8",
        "FC5",
        "FC1",
        "FC2",
        "FC6",
        "T7",
        "C3",
        "CZ",
        "C4",
        "T8",
        "CP5",
        "CP1",
        "CP2",
        "CP6",
        "P7",
        "P3",
        "PZ",
        "P4",
        "P8",
        "POZ",
        "O1",
        "OZ",
        "O2",
        "AF7",
        "AF3",
        "AF4",
        "AF8",
        "F5",
        "F1",
        "F2",
        "F6",
        "FC3",
        "FCZ",
        "FC4",
        "C5",
        "C1",
        "C2",
        "C6",
        "CP3",
        "CPZ",
        "CP4",
        "P5",
        "P1",
        "P2",
        "P6",
        "PO5",
        "PO3",
        "PO4",
        "PO6",
        "FT7",
        "FT8",
        "TP7",
        "TP8",
        "PO7",
        "PO8",
        "FT9",
        "FT10",
        "TPP9H",
        "TPP10H",
        "PO9",
        "PO10",
        "P9",
        "P10",
        "AFF1",
        "AFZ",
        "AFF2",
        "FFC5H",
        "FFC3H",
        "FFC4H",
        "FFC6H",
        "FCC5H",
        "FCC3H",
        "FCC4H",
        "FCC6H",
        "CCP5H",
        "CCP3H",
        "CCP4H",
        "CCP6H",
        "CPP5H",
        "CPP3H",
        "CPP4H",
        "CPP6H",
        "PPO1",
        "PPO2",
        "I1",
        "IZ",
        "I2",
        "AFP3H",
        "AFP4H",
        "AFF5H",
        "AFF6H",
        "FFT7H",
        "FFC1H",
        "FFC2H",
        "FFT8H",
        "FTT9H",
        "FTT7H",
        "FCC1H",
        "FCC2H",
        "FTT8H",
        "FTT10H",
        "TTP7H",
        "CCP1H",
        "CCP2H",
        "TTP8H",
        "TPP7H",
        "CPP1H",
        "CPP2H",
        "TPP8H",
        "PPO9H",
        "PPO5H",
        "PPO6H",
        "PPO10H",
        "POO9H",
        "POO3H",
        "POO4H",
        "POO10H",
        "OI1H",
        "OI2H",
    ],
)

Weibo2014 = moabb_wrapper(
    Weibo2014,
    200,
    [
        "FP1",
        "FPZ",
        "FP2",
        "AF3",
        "AF4",
        "F7",
        "F5",
        "F3",
        "F1",
        "Fz",
        "F2",
        "F4",
        "F6",
        "F8",
        "FT7",
        "FC5",
        "FC3",
        "FC1",
        "FCz",
        "FC2",
        "FC4",
        "FC6",
        "FT8",
        "T7",
        "C5",
        "C3",
        "C1",
        "CZ",
        "C2",
        "C4",
        "C6",
        "T8",
        "TP7",
        "CP5",
        "CP3",
        "CP1",
        "CPZ",
        "CP2",
        "CP4",
        "CP6",
        "TP8",
        "P7",
        "P5",
        "P3",
        "P1",
        "Pz",
        "P2",
        "P4",
        "P6",
        "P8",
        "PO7",
        "PO5",
        "PO3",
        "POZ",
        "PO4",
        "PO6",
        "PO8",
        "O1",
        "OZ",
        "O2",
    ],
)

# not available now
Zhou2016 = moabb_wrapper(
    Zhou2016,
    250,
    [
        "FP1",
        "FP2",
        "FC3",
        "FCZ",
        "FC4",
        "C3",
        "CZ",
        "C4",
        "CP3",
        "CPZ",
        "CP4",
        "O1",
        "OZ",
        "O2",
    ],
)

BNCI2014_002 = moabb_wrapper(BNCI2014_002, 512)

BNCI2015_001 = moabb_wrapper(BNCI2015_001, 512)

BNCI2015_004 = moabb_wrapper(BNCI2015_004, 256)

GrosseWentrup2009 = moabb_wrapper(GrosseWentrup2009, 500)

Lee2019_MI = moabb_wrapper(Lee2019_MI, 1000)

Ofner2017 = moabb_wrapper(Ofner2017, 512)

Liu2024 = moabb_wrapper(Liu2024, 500)

Shin2017A = moabb_wrapper(Shin2017A, 200)

Shin2017B = moabb_wrapper(Shin2017B, 200)

Stieger2021 = moabb_wrapper(Stieger2021, 1000)
