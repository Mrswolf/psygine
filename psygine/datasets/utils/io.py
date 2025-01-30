# -*- coding: utf-8 -*-
# Authors: swolf <swolfforever@gmail.com>
# Date: 2022/11/11
# License: MIT License
"""Basic methods related with IO.
"""
import os
import os.path as op
import json
import numpy as np
import scipy.io as sio
from scipy.io.matlab import mat_struct
import mat73
import mmap

def loadmat(filename):
    """Wrapper of scipy.io loadmat function, works for matv7.3 too.

    Parameters
    ----------
    filename : Union[str, Path]
        file path

    Returns
    -------
    dict
        data
    """    
    try:
        data = _loadmat(filename)
    except:
        data = mat73.loadmat(filename)
    return data

def _loadmat(filename):
    '''
    this function should be called instead of direct sio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects

    Notes: only works for mat before matlab v7.3
    '''
    def _check_keys(d):
        '''
        checks if entries in dictionary are mat-objects. If yes
        todict is called to change them to nested dictionaries
        '''
        for key in d:
            if isinstance(d[key], mat_struct):
                d[key] = _todict(d[key])
            elif isinstance(d[key], np.ndarray):
                d[key] = _tolist(d[key])
        return d

    def _todict(matobj):
        '''
        A recursive function which constructs from matobjects nested dictionaries
        '''
        d = {}
        for strg in matobj._fieldnames:
            elem = matobj.__dict__[strg]
            if isinstance(elem, mat_struct):
                d[strg] = _todict(elem)
            elif isinstance(elem, np.ndarray):
                d[strg] = _tolist(elem)
            else:
                d[strg] = elem
        return d

    def _tolist(ndarray):
        '''
        A recursive function which constructs lists from cellarrays
        (which are loaded as numpy ndarrays), recursing into the elements
        if they contain matobjects.
        '''
        if ndarray.dtype == object:
            elem_list = []
            for sub_elem in ndarray:
                if isinstance(sub_elem, mat_struct):
                    elem_list.append(_todict(sub_elem))
                elif isinstance(sub_elem, np.ndarray):
                    elem_list.append(_tolist(sub_elem))
                else:
                    elem_list.append(sub_elem)
            return elem_list
        else:
            return ndarray

    data = sio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)

def readjson(filename):
    with open(filename, 'r') as fid:
        try:
            config = json.load(fid)
        except ValueError:
            raise RuntimeError("not valid json file {:s}".format(filename))
    return config

def writejson(filename, content):
    with open(filename, 'w') as fid:
        json.dump(content, fid, sort_keys=True, indent=0)

def get_home_dir():
    home_dir = ''
    if 'nt' == os.name.lower():
        if op.isdir(op.join(os.getenv('APPDATA'), 'psygine')):
            home_dir = os.getenv('APPDATA')
        else:
            home_dir = os.getenv('USERPROFILE')
    else:
        home_dir = op.expanduser('~')
    return home_dir

def get_config_path():
    home_dir = get_home_dir()
    config_path = op.join(home_dir, 'psygine', 'psygine.json')
    return config_path

def get_config():
    config_path = get_config_path()
    if not op.isfile(config_path):
        config = {}
    else:
        config = readjson(config_path)
    return config

def get_local_path(key):
    config = get_config()
    
    if key in config:
        # get stored path
        path = config[key]
    else:
        # set default path
        path = op.join(get_home_dir(), 'psygine', 'psygine_data')
        if not op.exists(path):
            try:
                os.makedirs(path)
            except OSError:
                raise OSError("no permission to create default data folder")
    return path

def set_local_path(key, path):
    path = op.abspath(path)
    config = get_config()
    if path != config.get(key, ''):
        config[key] = path
        config_path = get_config_path()
        if not op.isfile(config_path):
            last_dir = op.dirname(config_path)
            os.makedirs(last_dir, exist_ok=True)
        writejson(config_path, config)
    return path

# The following code is adapted from bart 
# https://github.com/mrirecon/bart/tree/master/python/cfl.py

def readcfl(name):
    # get dims from .hdr
    with open(name + ".hdr", "rt") as h:
        h.readline() # skip
        l = h.readline()
    dims = [int(i) for i in l.split()]

    # remove singleton dimensions from the end
    n = np.prod(dims)
    dims_prod = np.cumprod(dims)
    dims = dims[:np.searchsorted(dims_prod, n)+1]

    # load data and reshape into dims
    with open(name + ".cfl", "rb") as d:
        a = np.fromfile(d, dtype=np.complex64, count=n);
    return a.reshape(dims, order='F') # column-major

def readmulticfl(name):
    # get dims from .hdr
    with open(name + ".hdr", "rt") as h:
        lines = h.read().splitlines()

    index_dim = 1 + lines.index('# Dimensions')
    total_size = int(lines[index_dim])
    index_sizes = 1 + lines.index('# SizesDimensions')
    sizes = [int(i) for i in lines[index_sizes].split()]
    index_dims = 1 + lines.index('# MultiDimensions')

    with open(name + ".cfl", "rb") as d:
        a = np.fromfile(d, dtype=np.complex64, count=total_size)

    offset = 0
    result = []
    for i in range(len(sizes)):
        dims = ([int(i) for i in lines[index_dims + i].split()])
        n = np.prod(dims)
        result.append(a[offset:offset+n].reshape(dims, order='F'))
        offset += n

    if total_size != offset:
        print("Error")

    return result

def writecfl(name, array):
    with open(name + ".hdr", "wt") as h:
        h.write('# Dimensions\n')
        for i in (array.shape):
                h.write("%d " % i)
        h.write('\n')

    size = np.prod(array.shape) * np.dtype(np.complex64).itemsize

    with open(name + ".cfl", "a+b") as d:
        os.ftruncate(d.fileno(), size)
        mm = mmap.mmap(d.fileno(), size, flags=mmap.MAP_SHARED, prot=mmap.PROT_WRITE)
        if array.dtype != np.complex64:
            array = array.astype(np.complex64)
        mm.write(np.ascontiguousarray(array.T))
        mm.close()
        #with mmap.mmap(d.fileno(), size, flags=mmap.MAP_SHARED, prot=mmap.PROT_WRITE) as mm:
        #    mm.write(array.astype(np.complex64).tobytes(order='F'))

def writemulticfl(name, arrays):
    size = 0
    dims = []

    for array in arrays:
        size += array.size
        dims.append(array.shape)

    with open(name + ".hdr", "wt") as h:
        h.write('# Dimensions\n')
        h.write("%d\n" % size)

        h.write('# SizesDimensions\n')
        for dim in dims:
            h.write("%d " % len(dim))
        h.write('\n')

        h.write('# MultiDimensions\n')
        for dim in dims:
            for i in dim:
                h.write("%d " % i)
            h.write('\n')
            
    size = size * np.dtype(np.complex64).itemsize

    with open(name + ".cfl", "a+b") as d:
        os.ftruncate(d.fileno(), size)
        mm = mmap.mmap(d.fileno(), size, flags=mmap.MAP_SHARED, prot=mmap.PROT_WRITE)
        for array in arrays:
            if array.dtype != np.complex64:
                array = array.astype(np.complex64)
            mm.write(np.ascontiguousarray(array.T))
        mm.close()
