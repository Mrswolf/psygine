# -*- coding: utf-8 -*-
#
# Authors: Swolf <swolfforever@gmail.com>
# Date: 2020/12/30
# License: MIT License
"""
Physionet MI.
"""
from .base import BaseEEGDataset
from ..utils.network import get_data_path
import numpy as np
from mne.io import read_raw_edf
from mne.channels import make_standard_montage

# PHYSIONET_URL = 'http://www.physionet.org/pn4/eegmmidb/'
PHYSIONET_URL = "https://www.physionet.org/files/eegmmidb/1.0.0/"


class BasePhysionet(BaseEEGDataset):
    """Physionet Motor Imagery dataset.

    Physionet MI dataset: https://physionet.org/pn4/eegmmidb/

    This data set consists of over 1500 one- and two-minute EEG recordings,
    obtained from 109 volunteers.

    Subjects performed different motor/imagery tasks while 64-channel EEG were
    recorded using the BCI2000 system (http://www.bci2000.org).
    Each subject performed 14 experimental runs: two one-minute baseline runs
    (one with eyes open, one with eyes closed), and three two-minute runs of
    each of the four following tasks:

    1. A target appears on either the left or the right side of the screen.
       The subject opens and closes the corresponding fist until the target
       disappears. Then the subject relaxes.

    2. A target appears on either the left or the right side of the screen.
       The subject imagines opening and closing the corresponding fist until
       the target disappears. Then the subject relaxes.

    3. A target appears on either the top or the bottom of the screen.
       The subject opens and closes either both fists (if the target is on top)
       or both feet (if the target is on the bottom) until the target
       disappears. Then the subject relaxes.

    4. A target appears on either the top or the bottom of the screen.
       The subject imagines opening and closing either both fists
       (if the target is on top) or both feet (if the target is on the bottom)
       until the target disappears. Then the subject relaxes.

    parameters
    ----------

    imagined: bool (default True)
        if True, return runs corresponding to motor imagination.

    executed: bool (default False)
        if True, return runs corresponding to motor execution.

    references
    ----------

    .. [1] Schalk, G., McFarland, D.J., Hinterberger, T., Birbaumer, N. and
           Wolpaw, J.R., 2004. BCI2000: a general-purpose brain-computer
           interface (BCI) system. IEEE Transactions on biomedical engineering,
           51(6), pp.1034-1043.

    .. [2] Goldberger, A.L., Amaral, L.A., Glass, L., Hausdorff, J.M., Ivanov,
           P.C., Mark, R.G., Mietus, J.E., Moody, G.B., Peng, C.K., Stanley,
           H.E. and PhysioBank, P., PhysioNet: components of a new research
           resource for complex physiologic signals Circulation 2000 Volume
           101 Issue 23 pp. E215–E220.
    """

    _EVENTS = {
        "rest": (1, (0, 3)),
        "left_hand": (2, (0, 3)),
        "right_hand": (3, (0, 3)),
        "hands": (4, (0, 3)),
        "feet": (5, (0, 3)),
        # "eyes_open": (6, (0, 60)),
        # "eyes_close": (7, (0, 60))
    }

    _CHANNELS = [
        "FC5",
        "FC3",
        "FC1",
        "FCZ",
        "FC2",
        "FC4",
        "FC6",
        "C5",
        "C3",
        "C1",
        "CZ",
        "C2",
        "C4",
        "C6",
        "CP5",
        "CP3",
        "CP1",
        "CPZ",
        "CP2",
        "CP4",
        "CP6",
        "FP1",
        "FPZ",
        "FP2",
        "AF7",
        "AF3",
        "AFZ",
        "AF4",
        "AF8",
        "F7",
        "F5",
        "F3",
        "F1",
        "FZ",
        "F2",
        "F4",
        "F6",
        "F8",
        "FT7",
        "FT8",
        "T7",
        "T8",
        "T9",
        "T10",
        "TP7",
        "TP8",
        "P7",
        "P5",
        "P3",
        "P1",
        "PZ",
        "P2",
        "P4",
        "P6",
        "P8",
        "PO7",
        "PO3",
        "POZ",
        "PO4",
        "PO8",
        "O1",
        "OZ",
        "O2",
        "IZ",
    ]

    def __init__(self, paradigm, is_imagined=True, local_path=None):
        super().__init__(
            "eegbci",
            list(range(0, 109)),
            [paradigm],
            events=self._EVENTS,
            channels=self._CHANNELS,
            srate=160,
            local_path=local_path,
        )

        self.is_imagined = is_imagined
        self.baseline_runs = [1, 2]
        self.feet_runs = []
        self.hand_runs = []

        if self.is_imagined:
            self.feet_runs += [6, 10, 14]
            self.hand_runs += [4, 8, 12]
        else:
            self.feet_runs += [5, 9, 13]
            self.hand_runs += [3, 7, 11]

    def _data_path(self, subject_id, local_path=None, force_update=False, proxies=None):
        if subject_id not in self.subjects:
            raise (IndexError("Invalid subject id"))

        runs = self.baseline_runs + self.hand_runs + self.feet_runs

        dests = []
        for r in runs:
            base_url = "{u}S{s:03d}/S{s:03d}R{r:02d}.edf".format(
                u=PHYSIONET_URL, s=subject_id + 1, r=r
            )
            dests.append(
                get_data_path(
                    base_url,
                    self.uid,
                    path=local_path,
                    proxies=proxies,
                    force_update=force_update,
                )
            )
        return [dests]

    def _get_single_subject_data(self, subject_id):
        dests = self._data_path(subject_id)
        montage = make_standard_montage("standard_1005")
        montage.rename_channels(
            {ch_name: ch_name.upper() for ch_name in montage.ch_names}
        )
        # montage.ch_names = [ch_name.upper() for ch_name in montage.ch_names]

        sess = dict()
        for isess, run_dests in enumerate(dests):
            runs = dict()
            for irun, run_file in enumerate(run_dests):
                raw = read_raw_edf(run_file, preload=True)
                raw.rename_channels(lambda x: x.strip(".").upper())
                raw.set_montage(montage)

                # change event id
                ori_desc = np.copy(raw.annotations.description)
                if irun == 0:
                    raw.annotations.description[ori_desc == "T0"] = 6
                    raw.annotations.description[ori_desc != "T0"] = 0
                if irun == 1:
                    raw.annotations.description[ori_desc == "T0"] = 7
                    raw.annotations.description[ori_desc != "T0"] = 0

                if irun in [2, 3, 4]:
                    raw.annotations.description[ori_desc == "T0"] = 1
                    raw.annotations.description[ori_desc == "T1"] = 2
                    raw.annotations.description[ori_desc == "T2"] = 3
                if irun in [5, 6, 7]:
                    raw.annotations.description[ori_desc == "T0"] = 1
                    raw.annotations.description[ori_desc == "T1"] = 4
                    raw.annotations.description[ori_desc == "T2"] = 5

                runs["run_{:d}".format(irun)] = raw
            sess["session_{:d}".format(isess)] = runs
        return sess


class PhysionetMI(BasePhysionet):
    def __init__(self, local_path=None):
        super().__init__("mi-eeg", is_imagined=True, local_path=local_path)
