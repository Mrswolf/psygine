# -*- coding: utf-8 -*-
#
# Authors: Swolf <swolfforever@gmail.com>
# Date: 2025/01/23
# License: MIT License
"""OCMR Dataset.
"""
import os
import ismrmrd
import ismrmrd.xsd
import numpy as np
import pandas as pd
from ..base import BaseDataset
from ..utils.network import get_data_path

# ocmr hint: https://ocmr.s3.amazonaws.com/data/fs_0001_1_5T.h5
# https://ocmr.s3.amazonaws.com/ocmr_data_attributes.csv

OCMR_DATA_ATTR_URL = "https://ocmr.s3.amazonaws.com/ocmr_data_attributes.csv"
OCMR_DATA_URL = "https://ocmr.s3.amazonaws.com/data"


# copy from https://github.com/MRIOSU/OCMR/blob/master/Python/read_ocmr.py
def read_ocmr(filename):
    # Before running the code, install ismrmrd-python and ismrmrd-python-tools:
    #  https://github.com/ismrmrd/ismrmrd-python
    #  https://github.com/ismrmrd/ismrmrd-python-tools
    # Last modified: 06-12-2020 by Chong Chen (Chong.Chen@osumc.edu)
    #
    # Input:  *.h5 file name
    # Output: all_data    k-space data, orgnazide as {'kx'  'ky'  'kz'  'coil'  'phase'  'set'  'slice'  'rep'  'avg'}
    #         param  some parameters of the scan
    #

    # This is a function to read K-space from ISMRMD *.h5 data
    # Modifid by Chong Chen (Chong.Chen@osumc.edu) based on the python script
    # from https://github.com/ismrmrd/ismrmrd-python-tools/blob/master/recon_ismrmrd_dataset.py

    if not os.path.isfile(filename):
        print("%s is not a valid file" % filename)
        raise SystemExit
    dset = ismrmrd.Dataset(filename, "dataset", create_if_needed=False)
    header = ismrmrd.xsd.CreateFromDocument(dset.read_xml_header())
    enc = header.encoding[0]

    # Matrix size
    eNx = enc.encodedSpace.matrixSize.x
    # eNy = enc.encodedSpace.matrixSize.y
    eNz = enc.encodedSpace.matrixSize.z
    eNy = enc.encodingLimits.kspace_encoding_step_1.maximum + 1
    # no zero padding along Ny direction

    # Field of View
    eFOVx = enc.encodedSpace.fieldOfView_mm.x
    eFOVy = enc.encodedSpace.fieldOfView_mm.y
    eFOVz = enc.encodedSpace.fieldOfView_mm.z

    # Save the parameters
    param = dict()
    param["TRes"] = str(header.sequenceParameters.TR)
    param["FOV"] = [eFOVx, eFOVy, eFOVz]
    param["TE"] = str(header.sequenceParameters.TE)
    param["TI"] = str(header.sequenceParameters.TI)
    param["echo_spacing"] = str(header.sequenceParameters.echo_spacing)
    param["flipAngle_deg"] = str(header.sequenceParameters.flipAngle_deg)
    param["sequence_type"] = header.sequenceParameters.sequence_type

    # Read number of Slices, Reps, Contrasts, etc.
    nCoils = header.acquisitionSystemInformation.receiverChannels
    try:
        nSlices = enc.encodingLimits.slice.maximum + 1
    except:
        nSlices = 1

    try:
        nReps = enc.encodingLimits.repetition.maximum + 1
    except:
        nReps = 1

    try:
        nPhases = enc.encodingLimits.phase.maximum + 1
    except:
        nPhases = 1

    try:
        nSets = enc.encodingLimits.set.maximum + 1
    except:
        nSets = 1

    try:
        nAverage = enc.encodingLimits.average.maximum + 1
    except:
        nAverage = 1

    # TODO loop through the acquisitions looking for noise scans
    firstacq = 0
    for acqnum in range(dset.number_of_acquisitions()):
        acq = dset.read_acquisition(acqnum)

        # TODO: Currently ignoring noise scans
        if acq.isFlagSet(ismrmrd.ACQ_IS_NOISE_MEASUREMENT):
            # print("Found noise scan at acq ", acqnum)
            continue
        else:
            firstacq = acqnum
            print("Imaging acquisition starts acq ", acqnum)
            break

    # assymetry echo
    kx_prezp = 0
    acq_first = dset.read_acquisition(firstacq)
    if acq_first.center_sample * 2 < eNx:
        kx_prezp = eNx - acq_first.number_of_samples

    # Initialiaze a storage array
    param["kspace_dim"] = {"kx ky kz coil phase set slice rep avg"}
    all_data = np.zeros(
        (eNx, eNy, eNz, nCoils, nPhases, nSets, nSlices, nReps, nAverage),
        dtype=np.complex64,
    )

    # check if pilot tone (PT) is on
    pilottone = 0
    try:
        if header.userParameters.userParameterLong[3].name == "PilotTone":
            pilottone = header.userParameters.userParameterLong[3].value
    except:
        pilottone = 0

    if pilottone == 1:
        print(
            "Pilot Tone is on, discarding the first 3 and last 1 k-space point for each line"
        )

    # Loop through the rest of the acquisitions and stuff
    for acqnum in range(firstacq, dset.number_of_acquisitions()):
        acq = dset.read_acquisition(acqnum)
        if (
            pilottone == 1
        ):  # discard the first 3 and last 1 k-space point to exclude PT artifact
            acq.data[:, [0, 1, 2, acq.data.shape[1] - 1]] = 0

        # Stuff into the buffer
        y = acq.idx.kspace_encode_step_1
        z = acq.idx.kspace_encode_step_2
        phase = acq.idx.phase
        set = acq.idx.set
        slice = acq.idx.slice
        rep = acq.idx.repetition
        avg = acq.idx.average
        all_data[kx_prezp:, y, z, :, phase, set, slice, rep, avg] = np.transpose(
            acq.data
        )

    return all_data, param


class OCMR(BaseDataset):
    """OCMR Dataset.

    OCMR is an open-access multi-coil k-space dataset for cardiovascular magnetic resonance imaging.
    Following is a table of attributes in the dataset:

    scn: This attribute identifies the field strength and type of scanner. The value of ‘15avan’ se-
    lects datasets collected on 1.5T Siemens MAGNETOM Avanto; the value of ‘15sola’ selects
    datasets collected on 1.5T Siemens MAGNETOM Sola; the value of ‘30pris’ selects datasets
    collected on 3T Siemens MAGNETOM Prisma.

    smp: This attribute identifies different sampling patterns. The value of ‘fs’ selects fully sampled
    datasets; the value of ‘pse’ selects prospectively undersampled datasets with pseudo-random
    sampling; the value of ‘uni’ selects prospectively undersampled datasets with uniform un-
    dersampling.

    ech: This attribute identifies asymmetric readout or echo. The value of ‘asy’ selects datasets
    with asymmetric echo, while the value of ‘sym’ selects datasets with symmetric echo.

    dur: This attribute only applies to prospectively undersampled data and distinguishes long from
    short scans. The value of ‘lng’ selects datasets where the time dimension is at least 5 s long,
    while the value of ‘shr’ selects datasets where the time dimension is shorter than 5 s. The
    value of ‘all’ selects datasets regardless of the duration. All fully sampled datasets belong
    to ‘shr’.

    viw: This attribute defines the view of the imaging slice. The value of ‘sax’ selects datasets
    collected in the short-axis view, while the value of ‘lax’ selects datasets collected in the
    long-axis view.

    sli: This attribute distinguishes individual slices from stacks. The value of ‘ind’ selects datasets
    collected as individual slices, while the value of ‘stk’ selects short-axis/long-axis stacks.

    fov: This attribute defines the presence of spatial aliasing, i.e., when the field-of-view (FOV)
    is smaller than the spatial extent of the content. The value of ‘ali’ selects datasets with
    spatial aliasing, while the value of ‘noa’ selects datasets without aliasing. The value of
    ‘all’ selects datasets regardless of spatial aliasing. Note, for SENSE-based methods, datasets
    with aliasing would require the utilization of multiple sets of sensitivity maps for artifact-free
    reconstruction.

    sub: This attribute distinguishes patients from healthy volunteers. The value of ‘pat’ selects data
    collected from patients, while the value of ‘vol’ selects data collected from healthy volunteers.

    See https://arxiv.org/pdf/2008.03410 for more details.
    """

    def __init__(self, local_path=None):
        super().__init__("ocmr")
        self.local_path = local_path
        data_attribute_path = get_data_path(
            OCMR_DATA_ATTR_URL,
            self.uid,
            path=self.local_path,
            proxies=None,
            force_update=False,
        )
        self._data_attribute = pd.read_csv(data_attribute_path)
        # drop rows with empty file name
        self._data_attribute = self._data_attribute[
            self._data_attribute["file name"].notna()
        ].reset_index(drop=True)

    def get_filename(self, idx):
        return self._data_attribute.iloc[idx]["file name"]

    def _data_path(self, idx, local_path=None, force_update=False, proxies=None):
        filename = self.get_filename(idx)
        data_path = get_data_path(
            f"{OCMR_DATA_URL}/{filename}",
            self.uid,
            path=local_path,
            proxies=proxies,
            force_update=force_update,
        )

        return data_path

    def __len__(self):
        return len(self._data_attribute)

    def __getitem__(self, idx):
        data_path = self._data_path(idx, self.local_path)
        kspace, scanparams = read_ocmr(data_path)
        return kspace, scanparams
