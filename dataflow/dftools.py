#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: dftools.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import sys, os
from scipy.misc import imsave

from utils.utils import mkdir_p

def dump_dataset_images(ds, dirname, max_count=None, index=0):
    """ dump images to a folder
        index: the index of the image in a data point
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
