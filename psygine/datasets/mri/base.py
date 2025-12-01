# -*- coding: utf-8 -*-
#
# Authors: Swolf <swolfforever@gmail.com>
# Date: 2023/03/12
# License: MIT License
"""Base MR Dataset Design."""
from ..base import BaseDataset

import ctypes
import os.path as op
from abc import abstractmethod
import ismrmrd
import ismrmrd.xsd
from ismrmrd.xsd import (
    matrixSizeType,
    fieldOfViewMm,
    limitType,
    trajectoryType,
    trajectoryDescriptionType,
    parallelImagingType,
)
import numpy as np
import pandas as pd


class RawMRI:
    """Raw MRI data class.

    Parameters
    ----------
    file : str or ismrmrd.Dataset
        file path or ismrmrd.Dataset object

    References
    ----------
    See https://ismrmrd.readthedocs.io/en/latest/_downloads/73341b9f70b92954beebbe9a78ff536f/ismrmrd.xsd for header description.
    See https://ismrmrd.readthedocs.io/en/latest/mrd_raw_data.html for acquisition description.
    """

    def __init__(self, file):
        if isinstance(file, ismrmrd.Dataset):
            self._dataset = file
        elif isinstance(file, str):
            if not op.isfile(file):
                raise ValueError(f"Path {file} is not a file")
            else:
                try:
                    self._dataset = ismrmrd.Dataset(
                        file, "dataset", create_if_needed=False
                    )
                except:
                    raise ValueError(f"{file} is not a ismrmrd file")
        else:
            raise ValueError(
                f"file {file} should be a ismrmrd file path or ismrmrd.Dataset object"
            )
        self._header = ismrmrd.xsd.CreateFromDocument(self._dataset.read_xml_header())
        self._raw_kdata = None
        self._raw_kdata_traj = None
        self._raw_kdata_meta = None
        # TODO: add support for image and waveform data
        self._raw_imagedata = None
        self._raw_waveformdata = None

        self._preload()

    def _preload(self):
        if self._raw_kdata is None:
            self._raw_kdata = []
            self._raw_kdata_traj = []
            self._raw_kdata_meta = []
            for i in range(self._dataset.number_of_acquisitions()):
                acq = self._dataset.read_acquisition(i)
                self._raw_kdata.append(acq.data)
                self._raw_kdata_traj.append(acq.traj)
                self._raw_kdata_meta.append(self._acq_header_to_dict(acq.getHead()))
            # self._raw_kdata = np.array(self._raw_kdata)
            # self._raw_kdata_traj = np.array(self._raw_kdata_traj)
            # if self._raw_kdata_traj.size == 0:
            #     self._raw_kdata_traj = None
            self._raw_kdata_meta = pd.DataFrame(self._raw_kdata_meta)

    def _acq_header_to_dict(self, acqheader):
        """Convert acquisition header to dict."""
        acq_dict = {}
        for field in acqheader._fields_:
            value = getattr(acqheader, field[0])

            if isinstance(value, ctypes.Array):
                value = np.array(value)
            elif isinstance(value, ismrmrd.acquisition.EncodingCounters):
                idx_dict = self._acq_header_to_dict(value)
                acq_dict.update(idx_dict)
                continue
            acq_dict[field[0]] = value

        return acq_dict

    def _isFlagSet(self, flags, ISRMRMRD_FLAG):
        return (flags & (1 << (ISRMRMRD_FLAG - 1))) > 0

    def get_noisescan_mask(self):
        flags = self._raw_kdata_meta["flags"]
        return self._isFlagSet(flags, ismrmrd.ACQ_IS_NOISE_MEASUREMENT)

    def get_navigator_mask(self):
        flags = self._raw_kdata_meta["flags"]
        return self._isFlagSet(flags, ismrmrd.ACQ_IS_NAVIGATION_DATA)

    def get_phasecorr_mask(self):
        flags = self._raw_kdata_meta["flags"]
        return self._isFlagSet(flags, ismrmrd.ACQ_IS_PHASECORR_DATA)

    def get_dummyscan_mask(self):
        flags = self._raw_kdata_meta["flags"]
        return self._isFlagSet(flags, ismrmrd.ACQ_IS_DUMMYSCAN_DATA)

    def get_rtfeedback_mask(self):
        flags = self._raw_kdata_meta["flags"]
        return self._isFlagSet(flags, ismrmrd.ACQ_IS_RTFEEDBACK_DATA)

    def get_hpfeedback_mask(self):
        flags = self._raw_kdata_meta["flags"]
        return self._isFlagSet(flags, ismrmrd.ACQ_IS_HPFEEDBACK_DATA)

    def get_surfacecoilcorrectionscan_mask(self):
        flags = self._raw_kdata_meta["flags"]
        return self._isFlagSet(flags, ismrmrd.ACQ_IS_SURFACECOILCORRECTIONSCAN_DATA)

    def get_phase_stabilization_reference_mask(self):
        flags = self._raw_kdata_meta["flags"]
        return self._isFlagSet(flags, ismrmrd.ACQ_IS_PHASE_STABILIZATION_REFERENCE)

    def get_phase_stabilization_mask(self):
        flags = self._raw_kdata_meta["flags"]
        return self._isFlagSet(flags, ismrmrd.ACQ_IS_PHASE_STABILIZATION)

    def get_parallel_calibration_mask(self):
        flags = self._raw_kdata_meta["flags"]
        return self._isFlagSet(flags, ismrmrd.ACQ_IS_PARALLEL_CALIBRATION)

    def get_imaging_mask(self):
        bMask = np.ones(self._raw_kdata_meta.shape[0], dtype=bool)
        bMask &= ~self.get_noisescan_mask()
        bMask &= ~self.get_navigator_mask()
        bMask &= ~self.get_phasecorr_mask()
        bMask &= ~self.get_dummyscan_mask()
        bMask &= ~self.get_rtfeedback_mask()
        bMask &= ~self.get_hpfeedback_mask()
        bMask &= ~self.get_surfacecoilcorrectionscan_mask()
        bMask &= ~self.get_phase_stabilization_reference_mask()
        bMask &= ~self.get_phase_stabilization_mask()
        bMask &= ~self.get_parallel_calibration_mask()
        return bMask

    def get_mask(
        self,
        imagscan=True,
        noisescan=False,
        navigator=False,
        phasecorr=False,
        dummyscan=False,
        rtfeedback=False,
        hpfeedback=False,
        surfacecoilcorrectionscan=False,
        phase_stabilization_reference=False,
        phase_stabilization=False,
        parallel_calibration=False,
    ):
        bMask = np.zeros(self._raw_kdata_meta.shape[0], dtype=bool)
        if imagscan:
            bMask |= self.get_imaging_mask()
        if noisescan:
            bMask |= self.get_noisescan_mask()
        if navigator:
            bMask |= self.get_navigator_mask()
        if phasecorr:
            bMask |= self.get_phasecorr_mask()
        if dummyscan:
            bMask |= self.get_dummyscan_mask()
        if rtfeedback:
            bMask |= self.get_rtfeedback_mask()
        if hpfeedback:
            bMask |= self.get_hpfeedback_mask()
        if surfacecoilcorrectionscan:
            bMask |= self.get_surfacecoilcorrectionscan_mask()
        if phase_stabilization_reference:
            bMask |= self.get_phase_stabilization_reference_mask()
        if phase_stabilization:
            bMask |= self.get_phase_stabilization_mask()
        if parallel_calibration:
            bMask |= self.get_parallel_calibration_mask()
        return bMask

    def get_data(
        self,
        concat=True,
        imagscan=True,
        noisescan=False,
        navigator=False,
        phasecorr=False,
        dummyscan=False,
        rtfeedback=False,
        hpfeedback=False,
        surfacecoilcorrectionscan=False,
        phase_stabilization_reference=False,
        phase_stabilization=False,
        parallel_calibration=False,
    ):
        bMask = self.get_mask(
            imagscan=imagscan,
            noisescan=noisescan,
            navigator=navigator,
            phasecorr=phasecorr,
            dummyscan=dummyscan,
            rtfeedback=rtfeedback,
            hpfeedback=hpfeedback,
            surfacecoilcorrectionscan=surfacecoilcorrectionscan,
            phase_stabilization_reference=phase_stabilization_reference,
            phase_stabilization=phase_stabilization,
            parallel_calibration=parallel_calibration,
        )

        data = [self._raw_kdata[i] for i, selected in enumerate(bMask) if selected]
        if concat:
            data = np.array(data)
        meta = self._raw_kdata_meta[bMask].reset_index()
        return data, meta

    def get_protocol(self):
        return self._header

    def _format_ismrmrd_types(self, root, attrList):
        def _format_ismrmrd_type(name, attrValue):
            if isinstance(attrValue, (matrixSizeType, fieldOfViewMm)):
                fstr = f"{name}:{[attrValue.x, attrValue.y, attrValue.z]}"
            elif isinstance(attrValue, limitType):
                fstr = f"{name}:{attrValue.minimum}-{attrValue.maximum}({attrValue.center})"
            elif attrValue is None:
                fstr = f"{name}:"
            return fstr, attrValue

        return [_format_ismrmrd_type(attr, getattr(root, attr)) for attr in attrList]

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        encs = self._header.encoding
        desc_str = """
EncodingSpace:{i}
    EncodedSpace(x,y,z):
        {encStr}
    ReconSpace(x,y,z):
        {reconStr}
    EncodingLimits(min-max(center)):
        {encLimitsStr}
            """
        desc = ""
        for i, enc in enumerate(encs):
            encList = self._format_ismrmrd_types(
                enc.encodedSpace, ["matrixSize", "fieldOfView_mm"]
            )
            encStr = " ".join([fstr for fstr, value in encList])
            reconList = self._format_ismrmrd_types(
                enc.reconSpace, ["matrixSize", "fieldOfView_mm"]
            )
            reconStr = " ".join([fstr for fstr, value in reconList])
            encLimitsList = self._format_ismrmrd_types(
                enc.encodingLimits,
                [
                    "kspace_encoding_step_0",
                    "kspace_encoding_step_1",
                    "kspace_encoding_step_2",
                    "average",
                    "slice",
                    "contrast",
                    "phase",
                    "repetition",
                    "set",
                    "segment",
                    "user_0",
                    "user_1",
                    "user_2",
                    "user_3",
                    "user_4",
                    "user_5",
                    "user_6",
                    "user_7",
                ],
            )
            encLimitsStr = "\n        ".join(
                [fstr for fstr, value in encLimitsList if value is not None]
            )
            desc += desc_str.format(
                i=i,
                encStr=encStr,
                reconStr=reconStr,
                encLimitsStr=encLimitsStr,
            )
        return desc


class BaseMRIDataset(BaseDataset):
    """Base MRI dataset class.

    Parameters
    ----------
    uid : str
        Unique identifier for the dataset.
    """

    def __init__(self, uid, local_path=None):
        super().__init__(uid)
        self.local_path = local_path

    def __getitem__(self, idx):
        return self._get_single_subject_data(idx)

    @abstractmethod
    def _data_path(self, idx, local_path=None, force_update=False, proxies=None):
        pass

    @abstractmethod
    def _get_single_subject_data(self, idx):
        pass

    def download_all(self, local_path=None, force_update=False, proxies=None):
        """Download all data in the dataset."""
        if local_path is not None:
            self.local_path = local_path

        for idx in range(len(self)):
            self._data_path(
                idx,
                local_path=self.local_path,
                force_update=force_update,
                proxies=proxies,
            )
