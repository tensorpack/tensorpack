# -*- coding: UTF-8 -*-
# File: dftools.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import sys, os
from scipy.misc import imsave

from ..utils.fs import mkdir_p

# TODO name_func to write label?
def dump_dataset_images(ds, dirname, max_count=None, index=0):
    """ Dump images from a `DataFlow` to a directory.

        :param ds: a `DataFlow` instance.
        :param dirname: name of the directory.
        :param max_count: max number of images to dump
        :param index: the index of the image component in a data point.
    """
    mkdir_p(dirname)
    if max_count is None:
        max_count = sys.maxint
    for i, dp in enumerate(ds.get_data()):
        print i
        if i > max_count:
            return
        img = dp[index]
        imsave(os.path.join(dirname, "{}.jpg".format(i)), img)
