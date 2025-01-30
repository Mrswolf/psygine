# -*- coding: utf-8 -*-
#
# Authors: Swolf <swolfforever@gmail.com>
# Date: 2025/01/30
# License: MIT License
"""Otazo Dataset.
"""
import numpy as np
from ..base import BaseDataset
from ..utils.io import loadmat
from ..utils.network import get_data_path

OTAZO_CARDIAC_PERF_R8_URL = "https://web.eecs.umich.edu/~fessler/irt/reproduce/19/lin-19-edp/data/cardiac_perf_R8.mat"
OTAZO_CARDIAC_PERF_R6_URL = "https://web.eecs.umich.edu/~fessler/irt/reproduce/19/lin-19-edp/data/cardiac_cine_R6.mat"
OTAZO_CARDIAC_PERF_FULL_URL = "https://web.eecs.umich.edu/~fessler/irt/reproduce/19/lin-19-edp/data/cardiac_perf_full_single.mat"
OTAZO_ABDOMEN_DCE_GA_URL = "https://web.eecs.umich.edu/~fessler/irt/reproduce/19/lin-19-edp/data/abdomen_dce_ga.mat"
OTAZO_ABDOMEN_DCE_GA_V2_URL = "https://web.eecs.umich.edu/~fessler/irt/reproduce/19/lin-19-edp/data/abdomen_dce_ga_v2.mat"

class BaseCardiacPerfDataset(BaseDataset):
    def __init__(self, uid, url, local_path=None):
        super().__init__(uid)
        self.url = url
        self.local_path = local_path

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self._load_data(self._data_path(local_path=self.local_path))

    def _data_path(self, local_path=None, force_update=False, proxies=None):
        path = get_data_path(self.url, self.uid, path=local_path, proxies=proxies, force_update=force_update)
        return path
    
    def _load_data(self, path):
        data = loadmat(path)
        return data
    
class OtazoCardiacPerfR6(BaseCardiacPerfDataset):
    def __init__(self, local_path=None):
        super().__init__(
            'otazo', 
            OTAZO_CARDIAC_PERF_R6_URL, 
            local_path=local_path)

class OtazoCardiacPerfR8(BaseCardiacPerfDataset):
    def __init__(self, local_path=None):
        super().__init__(
            'otazo', 
            OTAZO_CARDIAC_PERF_R8_URL, 
            local_path=local_path)

class OtazoCardiacPerfFull(BaseCardiacPerfDataset):
    def __init__(self, local_path=None):
        super().__init__(
            'otazo', 
            OTAZO_CARDIAC_PERF_FULL_URL, 
            local_path=local_path)

class BaseAbdomenDceDataset(BaseDataset):
    def __init__(self, uid, url, local_path=None):
        super().__init__(uid)
        self.url = url
        self.local_path = local_path

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self._load_data(self._data_path(local_path=self.local_path))

    def _data_path(self, local_path=None, force_update=False, proxies=None):
        path = get_data_path(self.url, self.uid, path=local_path, proxies=proxies, force_update=force_update)
        return path
    
    def _load_data(self, path):
        data = loadmat(path)
        return data
    
class OtazoAbdomenDce(BaseAbdomenDceDataset):
    r"""Otazo Abdomen DCE dataset.
    
    The abdomen_dce_ga.mat file has these variables in it:

    b1         384x384x12            28311552  double    complex   
    k          384x600                3686400  double    complex   
    kdata      384x600x12            44236800  double    complex   
    w          384x600                1843200  double             

    real(k) and imag(k) are the kx and ky k-space sampling coordinates
    kx = 384*real(k) is 384 x 600 (real), which is 600 spokes with 384 samples each
    ky = 384*imag(k)
    plot(kx, ky, '.')
    w is density compensation factors (DCF)
    b1 is the sensitivity maps for the 12 coils
    kdata is the noisy GA k-space data for 12 coils

    The 2015 paper by Otazo et al. (doi 10.1002/mrm.25240) used 8 spokes
    per frame so 600/8 = 75 frames

    To apply density-compensated gridding of this data, do this in Matlab:

    load abdomen_dce_ga.mat
    N = 384
    % kx = N*real(k);
    % ky = N*imag(k);
    kx = N*imag(k);
    ky = -N*real(k); % fix orientation
    plot(kx, ky, '.'), axis square
    dx = 1/N
    om = 2*pi*dx*[kx(:) ky(:)]; % omega for nufft code (-pi to pi)
    minmax(om) % only goes to pi/2 so something might be awry here?
    plot(om(:,1), om(:,2), '.'), axis_pipi
    A = Gnufft(true(N,N), {om, [N N], [6 6], 2*[N N], [N N]/2, 'table', 2^10, 'minmax:kb'})
    % DCF gridding recon of 1st coil using *all* data (ignoring time):
    x1 = reshape(A' * col(w .* kdata(:,:,1)), N, N);
    im(x1)
    """
    def __init__(self, local_path=None):
        super().__init__(
            'otazo', 
            OTAZO_ABDOMEN_DCE_GA_URL, 
            local_path=local_path)
        
class OtazoAbdomenDceV2(BaseAbdomenDceDataset):
    r"""Otazo Abdomen DCE dataset V2.
    
    2019-10-26 email from Li Feng says that the sampling should be -0.5 to 0.5
    rather than from -0.25 to 0.25 as seen in the original MCNUFFT.m code that
    went with the Otazo et. al L+S paper.

    2019-10-29 email from Ricardo Otazo has this updated file
    abdomen_dce_ga_v2.mat with these variables in it:

    b1: [384x384x12 double]
    kdata: [768x600x12 double]
    k: [768x600 double]
    w: [768x600 double]

    The b1 maps in the two files are the same if you normalize the new one
    b1 = b1 / max(abs(b1(:)))
    The "k" data ranges from -0.499 to 0.499 in this new file.

    This new data file was not used in our 2019 L+S paper, but can be useful
    for other experiments going forward.

    To apply density-compensated gridding of this data, do this in Matlab:

    load abdomen_dce_ga_v2.mat
    N = 384
    kx = N*imag(k);
    ky = -N*real(k); % fix orientation
    plot(kx, ky, '.'), axis square
    dx = 1/N
    om = 2*pi*dx*[kx(:) ky(:)]; % omega for nufft code (-pi to pi)
    minmax(om) % -pi to pi, yay!
    plot(om(:,1), om(:,2), '.'), axis_pipi
    A = Gnufft(true(N,N), {om, [N N], [6 6], 2*[N N], [N N]/2, 'table', 2^10, 'minmax:kb'})
    % DCF gridding recon of 1st coil using *all* data (ignoring time):
    x1 = reshape(A' * col(w .* kdata(:,:,1)), N, N);
    im(x1)
    """
    def __init__(self, local_path=None):
        super().__init__(
            'otazo', 
            OTAZO_ABDOMEN_DCE_GA_V2_URL, 
            local_path=local_path) 


