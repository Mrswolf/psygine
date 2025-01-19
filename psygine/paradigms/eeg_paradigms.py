# -*- coding: utf-8 -*-
#
# Authors: Swolf <swolfforever@gmail.com>
# Date: 2022/12/18
# License: MIT License
"""EEG Paradigm Design.
"""
import time
from collections import OrderedDict
import numpy as np
import pandas as pd
import mne
from sklearn.preprocessing import LabelEncoder
from .base import BaseParadigm
from .utils.hooks import RemovableHandle

__all__ = ["BaseEegParadigm", "MiEegParadigm", "SsvepEegParadigm"]


class BaseEegParadigm(BaseParadigm):
    r"""Base EEG Paradigm.

    Parameters
    ----------
    uid : str
        The unique string id to identify the paradigm.
    channels : list, optional
        A list of selected channel names. If None, use all channels the dataset supported.
    events : list, optional
        A list of selected event names. If None, use all events the dataset supported.
    intervals : list, optional
        A list of event intervals corresponding to selected events.
        e.g. [[-1, 1]] means epoching raw data from -1s to 1s for the first event.
        If None, use default settings from the dataset.
    srate : float or int, optional
        Resampling rate. If None, use default sampling rate.
    dataset_hooks : bool
        Whether to use default hooks from the dataset, default True.
        However, any user defined hook would override the dataset default hook.
    label_transform : bool
        Whether to encode labels with value between 0 and n_classes-1, default False.
    Attributes
    ----------
    uid : str
        The unique id for the current paradigm.
    """

    def __init__(
        self,
        uid,
        channels=None,
        events=None,
        intervals=None,
        srate=None,
        dataset_hooks=True,
        label_transform=False  
    ):
        super().__init__(uid)
        if channels is not None:
            channels = [ch.upper() for ch in channels]
        self._paradigm_channels = channels
        self._paradigm_events = events
        self._paradigm_intervals = intervals
        self._paradigm_srate = srate
        self._dataset_hooks = dataset_hooks
        self.label_transform = label_transform
        self._raw_hooks = OrderedDict()
        self._epoch_hooks = OrderedDict()
        self._array_hooks = OrderedDict()

    def _is_valid_args(self, dataset, channels, events, intervals, srate):
        if channels is not None:
            for channel in channels:
                if channel not in dataset.channels:
                    raise ValueError(
                        "{:s} is not an available channel in dataset {:s}.".format(
                            channel, dataset.uid
                        )
                    )

        if events is not None:
            for event in events:
                if event not in dataset.events:
                    raise ValueError(
                        "{:s} is not an available event in dataset {:s}.".format(
                            event, dataset.uid
                        )
                    )

        if intervals is not None:
            if 1 != len(intervals) and len(events) != len(intervals):
                return False

            for interval in intervals:
                if interval[0] >= interval[1]:
                    raise ValueError("Invalid interval:{}.".format(interval))

        if srate is not None:
            if not isinstance(srate, (int, float)):
                return False
        return True

    def is_valid(self, dataset):
        r"""check the validity of the EEG dataset.

        Parameters
        ----------
        dataset : BaseEegDataset
            An instance of BaseEegDataset.

        Returns
        -------
        bool
            Return True iff the dataset is valid, else False.
        """
        # check paradigms
        if self.uid not in dataset.paradigms:
            return False
        # check arguments
        if not self._is_valid_args(
            dataset,
            self._paradigm_channels,
            self._paradigm_events,
            self._paradigm_intervals,
            self._paradigm_srate,
        ):
            return False
        return True

    def _parse_args(self, dataset, channels, events, intervals, srate):
        # check srate
        if srate is None:
            srate = dataset.srate
        # check channels
        if channels is None:
            channels = dataset.channels
        # check events
        if events is None:
            events = dataset.events
        # check intervals
        if intervals is None:
            intervals = [dataset.get_event_interval(event) for event in events]
        else:
            if 1 == len(intervals):
                intervals = intervals * len(events)
        return channels, events, intervals, srate

    def _get_single_subject_data(self, dataset, subject_id):
        channels, events, intervals, srate = self._parse_args(
            dataset,
            self._paradigm_channels,
            self._paradigm_events,
            self._paradigm_intervals,
            self._paradigm_srate,
        )

        if self.label_transform:
            le = LabelEncoder().fit(events)

        rawdata = dataset.get_rawdata(subject_ids=[subject_id])[
            "{:d}".format(subject_id)
        ]

        sub_X, sub_y, sub_meta = {}, {}, {}
        for session_id, runs in rawdata.items():
            for run_id, raw in runs.items():
                if self._raw_hooks:
                    for hook in (*self._raw_hooks.values(),):
                        result = hook(self, raw)
                        if result is not None:
                            if not isinstance(result, tuple):
                                result = (result,)
                            (raw,) = result
                elif self._dataset_hooks:
                    if hasattr(dataset, "raw_hook"):
                        result = dataset.raw_hook(self, raw)
                        if result is not None:
                            if not isinstance(result, tuple):
                                result = (result,)
                            (raw,) = result

                channel_picks = mne.pick_channels(
                    dataset.channels, channels, ordered=True
                )

                # find available events, first check stim channels then annotations
                try:
                    all_events = mne.find_events(
                        raw, shortest_event=0, initial_event=True, verbose=False
                    )
                except ValueError:
                    all_events, _ = mne.events_from_annotations(
                        raw, event_id=lambda x: int(x), verbose=False
                    )

                for event, interval in zip(events, intervals):
                    try:
                        epoch = mne.Epochs(
                            raw,
                            all_events,
                            event_id={event: dataset.get_event_id(event)},
                            tmin=interval[0],
                            tmax=interval[1]
                            - 1.0 / raw.info["sfreq"],  # exclude the end point
                            picks=channel_picks,
                            proj=False,
                            preload=True,
                            baseline=None,
                            on_missing="raise",
                            event_repeated="drop",
                            verbose=False,
                        )
                    except ValueError:
                        # skip the empty event
                        continue

                    if self._epoch_hooks:
                        for hook in (*self._epoch_hooks.values(),):
                            result = hook(self, epoch)
                            if result is not None:
                                if not isinstance(result, tuple):
                                    result = (result,)
                                (epoch,) = result
                    elif self._dataset_hooks:
                        if hasattr(dataset, "epoch_hook"):
                            result = dataset.epoch_hook(self, epoch)
                            if result is not None:
                                if not isinstance(result, tuple):
                                    result = (result,)
                                (epoch,) = result

                    if srate < dataset.srate:
                        epoch = epoch.resample(srate, verbose=False)

                    X = epoch[event].get_data() * 1e6  # default micro-volt
                    if self.label_transform:
                        y = le.transform([event] * len(X))
                    else:
                        y = np.ones((len(X)), np.int64) * dataset.get_event_id(event)
                    meta = pd.DataFrame(
                        {
                            "subject": ["{:d}".format(subject_id)] * len(X),
                            "session": [session_id] * len(X),
                            "run": [run_id] * len(X),
                            "trial": epoch[event].selection,
                            "event": [event] * len(X),
                            "event_id": y,
                            "dataset": [dataset.uid] * len(X),
                        }
                    )

                    if self._array_hooks:
                        for hook in (*self._array_hooks.values(),):
                            result = hook(self, X, y, meta)
                            if result is not None:
                                if not isinstance(result, tuple):
                                    result = (result,)
                                X, y, meta = result
                    elif self._dataset_hooks:
                        if hasattr(dataset, "array_hook"):
                            result = dataset.array_hook(self, X, y, meta)
                            if result is not None:
                                if not isinstance(result, tuple):
                                    result = (result,)
                                X, y, meta = result

                    # gathering data
                    sub_X[event] = (
                        np.concatenate((sub_X[event], X), axis=0)
                        if event in sub_X
                        else X
                    )
                    sub_y[event] = (
                        np.concatenate((sub_y[event], y), axis=0)
                        if event in sub_y
                        else y
                    )
                    sub_meta[event] = (
                        pd.concat((sub_meta[event], meta), axis=0, ignore_index=True)
                        if event in sub_meta
                        else meta
                    )
        return sub_X, sub_y, sub_meta

    def get_data(self, dataset, subject_ids=None, n_jobs=None, concat=False):
        r"""Get data from multiple subjects.

        Parameters
        ----------
        dataset : BaseEegDataset
            An instance of BaseEegDataset.
        subject_ids : list, optional
            A list of queried subject ids. If None, use all subjects the dataset supported.
        n_jobs : int, optional
            The number of cores to use to load data, -1 for all cores.
        concat : bool
            Return concated data if True.

        Returns
        -------
        X : dict or array_like
            A dictionary of data, where keys are event names.
            Return concated numpy array if concat is True.
        y : dict or array_like
            A dictionary of label, where keys are event names.
            Return concated numpy array if concat is True.
        meta : dict or dataframe
            A dictionary of meta, where keys are event names.
            Return concated pandas dataframe if concat is True.
        """
        st = time.time()
        X_list, y_list, meta_list = super().get_data(dataset, subject_ids, n_jobs)
        _, events, _, _ = self._parse_args(
            dataset,
            self._paradigm_channels,
            self._paradigm_events,
            self._paradigm_intervals,
            self._paradigm_srate,
        )
        if subject_ids is None:
            subject_ids = dataset.subjects
        X, y, meta = {}, {}, {}
        for event in events:
            X[event] = np.concatenate(
                [
                    X_list[i][event]
                    for i in range(len(subject_ids))
                    if event in X_list[i]
                ],
                axis=0,
            )
            y[event] = np.concatenate(
                [
                    y_list[i][event]
                    for i in range(len(subject_ids))
                    if event in y_list[i]
                ],
                axis=0,
            )
            meta[event] = pd.concat(
                [
                    meta_list[i][event]
                    for i in range(len(subject_ids))
                    if event in meta_list[i]
                ],
                axis=0,
                ignore_index=True,
            )
        if concat:
            try:
                X = np.concatenate([X[event] for event in events], axis=0)
                y = np.concatenate([y[event] for event in events], axis=0)
                meta = pd.concat(
                    [meta[event] for event in events], axis=0, ignore_index=True
                )
            except ValueError:
                print(
                    "All the input array dimensions for the concatenation axis must match exactly.\nSkip concat step."
                )
        elapsed_time = time.time() - st
        print("Loading time: {:.4f}s".format(elapsed_time))
        return X, y, meta

    def register_raw_hook(self, hook):
        handle = RemovableHandle(self._raw_hooks)
        self._raw_hooks[handle.id] = hook
        return handle

    def register_epoch_hook(self, hook):
        handle = RemovableHandle(self._epoch_hooks)
        self._epoch_hooks[handle.id] = hook
        return handle

    def register_array_hook(self, hook):
        handle = RemovableHandle(self._array_hooks)
        self._array_hooks[handle.id] = hook
        return handle


class MiEegParadigm(BaseEegParadigm):
    r"""Base MI EEG Paradigm.

    Parameters
    ----------
    channels : list, optional
        A list of selected channel names. If None, use all channels the dataset supported.
    events : list, optional
        A list of selected event names. If None, use all events the dataset supported.
    intervals : list, optional
        A list of event intervals corresponding to selected events.
        e.g. [[-1, 1]] means epoching raw data from -1s to 1s for the first event.
        If None, use default settings from the dataset.
    srate : float or int, optional
        Resampling rate. If None, use default sampling rate.
    dataset_hooks : bool
        Whether to use default hooks from the dataset, default True.
        However, any user defined hook would override the dataset default hook.

    Attributes
    ----------
    uid : str
        The unique id for the current paradigm.
    """

    def __init__(
        self, channels=None, events=None, intervals=None, srate=None, dataset_hooks=True
    ):
        super().__init__(
            "mi-eeg",
            channels=channels,
            events=events,
            intervals=intervals,
            srate=srate,
            dataset_hooks=dataset_hooks,
            label_transform=True
        )


class SsvepEegParadigm(BaseEegParadigm):
    r"""Base SSVEP EEG Paradigm.

    Parameters
    ----------
    channels : list, optional
        A list of selected channel names. If None, use all channels the dataset supported.
    events : list, optional
        A list of selected event names. If None, use all events the dataset supported.
    intervals : list, optional
        A list of event intervals corresponding to selected events.
        e.g. [[-1, 1]] means epoching raw data from -1s to 1s for the first event.
        If None, use default settings from the dataset.
    srate : float or int, optional
        Resampling rate. If None, use default sampling rate.
    dataset_hooks : bool
        Whether to use default hooks from the dataset, default True.
        However, any user defined hook would override the dataset default hook.

    Attributes
    ----------
    uid : str
        The unique id for the current paradigm.
    """

    def __init__(
        self, channels=None, events=None, intervals=None, srate=None, dataset_hooks=True
    ):
        super().__init__(
            "ssvep-eeg",
            channels=channels,
            events=events,
            intervals=intervals,
            srate=srate,
            dataset_hooks=dataset_hooks,
            label_transform=True
        )
