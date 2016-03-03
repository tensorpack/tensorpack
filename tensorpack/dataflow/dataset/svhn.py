#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: svhn.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import os
import random
import numpy
import scipy
import scipy.io
from six.moves import range

from ...utils import logger
from ..base import DataFlow

__all__ = ['SVHNDigit']

class SVHNDigit(DataFlow):
    """
    SVHN Cropped Digit Dataset
    return img of 32x32x3, label of 0-9
    """
    def __init__(self, name, data_dir=None):
        """
        name: 'train', 'test', or 'extra'
        """
        if data_dir is None:
            data_dir = os.path.join(
                os.path.dirname(__file__),
                'svhn_data')
        assert name in ['train', 'test', 'extra'], name
        filename = os.path.join(data_dir, name + '_32x32.mat')
        assert os.path.isfile(filename), \
                "File {} not found! Download it from \
http://ufldl.stanford.edu/housenumbers/".format(filename)
        logger.info("Loading {} ...".format(filename))
        data = scipy.io.loadmat(filename)
        self.X = data['X'].transpose(3,0,1,2)
        self.Y = data['y'].reshape((-1))
        self.Y[self.Y==10] = 0

    def size(self):
        return self.X.shape[0]

    def get_data(self):
        n = self.X.shape[0]
        for k in range(n):
            yield [self.X[k], self.Y[k]]

