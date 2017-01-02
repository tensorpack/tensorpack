# -*- coding: UTF-8 -*-
# File: utils.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import os
import sys
from contextlib import contextmanager
import inspect
from datetime import datetime
from tqdm import tqdm
import numpy as np

__all__ = ['change_env',
           'get_rng',
           'get_dataset_path',
           'get_tqdm_kwargs',
           'get_tqdm',
           'execute_only_once',
           'building_rtfd',
           'intensity_to_rgb',
           'filter_intensity'
           ]


@contextmanager
def change_env(name, val):
    oldval = os.environ.get(name, None)
    os.environ[name] = val
    yield
    if oldval is None:
        del os.environ[name]
    else:
        os.environ[name] = oldval


def get_rng(obj=None):
    """ obj: some object to use to generate random seed"""
    seed = (id(obj) + os.getpid() +
            int(datetime.now().strftime("%Y%m%d%H%M%S%f"))) % 4294967295
    return np.random.RandomState(seed)


_EXECUTE_HISTORY = set()


def execute_only_once():
    """
    when called with:
        if execute_only_once():
            # do something
    The body is guranteed to be executed only the first time.
    """
    f = inspect.currentframe().f_back
    ident = (f.f_code.co_filename, f.f_lineno)
    if ident in _EXECUTE_HISTORY:
        return False
    _EXECUTE_HISTORY.add(ident)
    return True


def get_dataset_path(*args):
    d = os.environ.get('TENSORPACK_DATASET', None)
    if d is None:
        d = os.path.abspath(os.path.join(
            os.path.dirname(__file__), '..', 'dataflow', 'dataset'))
        if execute_only_once():
            from . import logger
            logger.info("TENSORPACK_DATASET not set, using {} for dataset.".format(d))
    assert os.path.isdir(d), d
    return os.path.join(d, *args)


def get_tqdm_kwargs(**kwargs):
    default = dict(
        smoothing=0.5,
        dynamic_ncols=True,
        ascii=True,
        bar_format='{l_bar}{bar}|{n_fmt}/{total_fmt}[{elapsed}<{remaining},{rate_noinv_fmt}]'
    )
    f = kwargs.get('file', sys.stderr)
    if f.isatty():
        default['mininterval'] = 0.5
    else:
        default['mininterval'] = 60
    default.update(kwargs)
    return default


def get_tqdm(**kwargs):
    return tqdm(**get_tqdm_kwargs(**kwargs))


def building_rtfd():
    return os.environ.get('READTHEDOCS') == 'True' \
        or os.environ.get('TENSORPACK_DOC_BUILDING')


def intensity_to_rgb(intensity, cmap='Blues_r'):
    """Convert a matrix of intensities to an rgb image employing a colormap.

    Nice colormaps are:
      - Blues, Blues_r
      - Reds, Reds_r
      - BuGns, BuGns_r
      - Greys, Greys_r

    Args:
        intensity (TYPE): array of intensities such as saliency
        background (None, optional): background for heatmap
        cmap (str, optional): used colormap (required matplotlib)

    Returns:
        TYPE: nice heatmap
    """
    def rgb2gray(rgb):
        r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray

    if not len(intensity.shape) == 2:
        intensity = rgb2gray(intensity)

    intensity = intensity.astype("float")
    intensity -= intensity.min()
    intensity /= intensity.max()

    import matplotlib.pyplot as plt

    cmap = plt.get_cmap(cmap)
    intensity = cmap(intensity.flatten())[..., 0:3].reshape([intensity.shape[0], intensity.shape[1], 3])
    intensity = intensity[:, :, [2, 1, 0]]
    return intensity


def filter_intensity(intensity, rgb):
    """Only highlight parts having high intensity values

    Args:
        intensity (TYPE): importance of specific pixel
        rgb (TYPE): original image

    Returns:
        TYPE: image with attention
    """
    assert intensity.shape[:2] == rgb.shape[:2]

    intensity = intensity.astype("float")
    intensity -= intensity.min()
    intensity /= intensity.max()
    intensity = 1 - intensity

    gray = rgb * 0 + 255 // 2

    return intensity[:, :, None] * gray + (1 - intensity[:, :, None]) * rgb
