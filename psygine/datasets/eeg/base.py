# -*- coding: utf-8 -*-
#
# Authors: Swolf <swolfforever@gmail.com>
# Date: 2022/12/01
# License: MIT License
"""Base EEG Dataset Design.
"""
from abc import abstractmethod

from ..base import BaseDataset


class BaseEegDataset(BaseDataset):
    r"""Base EEG Dataset.

    Parameters
    ----------
    uid : str
        The unique string id to identify the dataset.
    subjects : list
        A list of available subject ids.
    paradigms : list
        A list of valid paradigm uids.
    events : dict
        A dictionary containing all available events, including ids and intervals.
    channels : list
        A list of available channel names.
    srate : int or float
        The sampling rate of the dataset.

    Attributes
    ----------
    uid : str
        The unique id for the current dataset.
    subjects : list
        All available subject ids.
    paradigms : list
        All valid paradigm uids.
    events : list
        All available event names.
    intervals : list
        All available event intervals.
    event_ids : list
        All available event ids.
    channels : list
        All available channel names.
    srate : int or float
        The sampling rate of the dataset.
    """

    def __init__(self, uid, subjects, paradigms, events, channels, srate):
        super().__init__(uid)
        self._subject_ids = subjects
        self._valid_paradigms = paradigms
        self._dataset_events = events
        self._dataset_channels = [ch.upper() for ch in channels]
        self._srate = srate

    @property
    def subjects(self):
        return self._subject_ids

    @property
    def paradigms(self):
        return self._valid_paradigms

    @property
    def events(self):
        return list(self._dataset_events.keys())

    @property
    def intervals(self):
        return [self.get_event_interval(event) for event in self.events]

    @property
    def event_ids(self):
        return [self.get_event_id(event) for event in self.events]

    @property
    def channels(self):
        return self._dataset_channels

    @property
    def srate(self):
        return self._srate

    @abstractmethod
    def data_path(self, subject_id, local_path=None, force_update=False, proxies=None):
        r"""Mapping remote data to local and return the local file path.

        Parameters
        ----------
        subject_id : int
            The subject id.
        local_path : str, optional
            The local path to store remote data.
            If None, the default path is the psygine_data folder under the home directory.
        force_update : bool
            Whether to force update local files, default False.
        proxies : dict
            Proxy settings from the Request package.
        """

    @abstractmethod
    def _get_single_subject_data(self, subject_id):
        r"""Get a single subject's raw data.

        Parameters
        ----------
        subject_id : int
            The subject id.

        Returns
        -------
        dict
            A dictionary containing raw data, which are structured in session->run order.
        """

    def get_rawdata(self, subject_ids=None):
        r"""Get raw data from multiple subjects.

        Parameters
        ----------
        subject_ids : list, optional
            A list of selected subject ids. If None, use all available subjects in the dataset.

        Returns
        -------
        dict
            A dictionary containing raw data, which are structured in subject->session->run order.
        """
        if subject_ids is None:
            # use all subjects if not provided
            subject_ids = self.subjects
        else:
            # else check if the subject is valid
            for subject_id in subject_ids:
                if subject_id not in self.subjects:
                    raise ValueError("Invalid subject {}.".format(subject_id))

        rawdata = dict()
        for subject_id in subject_ids:
            rawdata["subject_{:d}".format(subject_id)] = self._get_single_subject_data(
                subject_id
            )
        return rawdata

    def download_all(self, local_path=None, force_update=False, proxies=None):
        r"""Download all subjects' data.

        parameters
        ----------
        local_path : str, optional
            The local path to store remote data.
            If None, the default path is the psygine_data folder under the home directory.
        force_update : bool
            Whether to force update local files, default False.
        proxies : dict
            Proxy settings from the Request package.
        """
        for subject_id in self.subjects:
            self.data_path(
                subject_id,
                local_path=local_path,
                force_update=force_update,
                proxies=proxies,
            )

    def get_event_id(self, event):
        r"""Get event id.

        Parameters
        ----------
        event : str
            Event name.

        Returns
        -------
        int
            Event id.
        """
        return self._dataset_events[event][0]

    def get_event_interval(self, event):
        r"""Get event interval.

        Parameters
        ----------
        event : str
            Event name.

        Returns
        -------
        list
            A list containing start time and end time.
        """
        return self._dataset_events[event][1]

    def __str__(self):
        event_info = "\n".join(
            [
                "    {}: {}".format(event, self._dataset_events[event])
                for event in self.events
            ]
        )
        desc = """Dataset {:s}:\n  Subjects  {:d}\n  Srate     {:.1f}\n  Events   \n{}\n  Channels  {:d}\n""".format(
            self.uid, len(self.subjects), self.srate, event_info, len(self.channels)
        )
        return desc

    def __repr__(self):
        return self.__str__()


class SsvepEegDataset(BaseEegDataset):
    r"""SSVEP EEG Dataset.

    Parameters
    ----------
    uid : str
        The unique string id to identify the dataset.
    subjects : list
        A list of available subject ids.
    events : dict
        A dictionary containing all available events, including ids and intervals.
    channels : list
        A list of available channel names.
    srate : int or float
        The sampling rate of the dataset.
    freq_phase_table : dict
        A dictionary containing frequencies and phases.

    Attributes
    ----------
    uid : str
        The unique id for the current dataset.
    subjects : list
        All available subject ids.
    paradigms : list
        All valid paradigm uids.
    events : list
        All available event names.
    intervals : list
        All available event intervals.
    event_ids : list
        All available event ids.
    channels : list
        All available channel names.
    srate : int or float
        The sampling rate of the dataset.
    """

    def __init__(self, uid, subjects, events, channels, srate, freq_phase_table):
        super().__init__(uid, subjects, ["ssvep-eeg"], events, channels, srate)
        self._freq_phase_table = freq_phase_table

    def get_event_frequency(self, event):
        r"""Get event frequency.

        Parameters
        ----------
        event : str
            Event name.

        Returns
        -------
        float
            Stimuli frequency.
        """
        return self._freq_phase_table[event][0]

    def get_event_phase(self, event):
        r"""Get event phase.

        Parameters
        ----------
        event : str
            Event name.

        Returns
        -------
        float
            Stimuli phase.
        """
        return self._freq_phase_table[event][1]


class MiEegDataset(BaseEegDataset):
    r"""MI EEG Dataset.

    Parameters
    ----------
    uid : str
        The unique string id to identify the dataset.
    subjects : list
        A list of available subject ids.
    events : dict
        A dictionary containing all available events, including ids and intervals.
    channels : list
        A list of available channel names.
    srate : int or float
        The sampling rate of the dataset.
    freq_phase_table : dict
        A dictionary containing frequencies and phases.

    Attributes
    ----------
    uid : str
        The unique id for the current dataset.
    subjects : list
        All available subject ids.
    paradigms : list
        All valid paradigm uids.
    events : list
        All available event names.
    intervals : list
        All available event intervals.
    event_ids : list
        All available event ids.
    channels : list
        All available channel names.
    srate : int or float
        The sampling rate of the dataset.
    """

    def __init__(self, uid, subjects, events, channels, srate):
        super().__init__(uid, subjects, ["mi-eeg"], events, channels, srate)
