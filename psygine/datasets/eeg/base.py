# -*- coding: utf-8 -*-
#
# Authors: Swolf <swolfforever@gmail.com>
# Date: 2022/12/01
# License: MIT License
"""Base EEG Dataset Design.
"""
from ..base import BaseDataset

class EegDataset(BaseDataset):
    def __init__(self,
        uid,
        subjects,
        paradigms,
        events,
        channels,
        srate):
        super().__init__(uid, subjects, paradigms)
        self._dataset_events = events
        self._dataset_channels = [ch.upper() for ch in channels]
        self._srate = srate

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

    def get_event_id(self, event):
        return self._dataset_events[event][0]

    def get_event_interval(self, event):
        return self._dataset_events[event][1]

    def __str__(self):
        event_info = '\n'.join(["    {}: {}".format(event, self._dataset_events[event]) for event in self.events])
        desc = """Dataset {:s}:\n  Subjects  {:d}\n  Srate     {:.1f}\n  Events   \n{}\n  Channels  {:d}\n""".format(
            self.uid, 
            len(self.subjects), 
            self.srate, 
            event_info, 
            len(self.channels)
        )
        return desc
    
    def __repr__(self):
        return self.__str__()

class SsvepEegDataset(EegDataset):
    def __init__(self,
        uid,
        subjects,
        events,
        channels,
        srate,
        freq_phase_table):
        super().__init__(uid, subjects, ['ssvep-eeg'], events, channels, srate)
        self._freq_phase_table = freq_phase_table

    def get_event_frequency(self, event):
        return self._freq_phase_table[event][0]

    def get_event_phase(self, event):
        return self._freq_phase_table[event][1]

class MiEegDataset(EegDataset):
    def __init__(self,
        uid,
        subjects,
        events,
        channels,
        srate):
        super().__init__(uid, subjects, ['mi-eeg'], events, channels, srate)