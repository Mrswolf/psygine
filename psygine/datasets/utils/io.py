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
import mat73

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
            if isinstance(d[key], sio.matlab.mio5_params.mat_struct):
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
            if isinstance(elem, sio.matlab.mio5_params.mat_struct):
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
        if ndarray.dtype == np.object:
            elem_list = []
            for sub_elem in ndarray:
                if isinstance(sub_elem, sio.matlab.mio5_params.mat_struct):
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
                os.mkdir(path)
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

    
    