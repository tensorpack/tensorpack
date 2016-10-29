#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# File: svhn.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import os
import random
import numpy as np
from six.moves import range

from ...utils import logger, get_rng, get_dataset_path
from ..base import RNGDataFlow

try:
    import scipy.io
    __all__ = ['SVHNDigit']
except ImportError:
    logger.warn("Cannot import scipy. SVHNDigit dataset won't be available!")
    __all__ = []

SVHN_URL = "http://ufldl.stanford.edu/housenumbers/"

class SVHNDigit(RNGDataFlow):
    """
    SVHN Cropped Digit Dataset.
    return img of 32x32x3, label of 0-9
    """
    _Cache = {}

    def __init__(self, name, data_dir=None, shuffle=True):
        """
        :param name: 'train', 'test', or 'extra'
        :param data_dir: a directory containing the original {train,test,extra}_32x32.mat
        """
        self.shuffle = shuffle

        if name in SVHNDigit._Cache:
            self.X, self.Y = SVHNDigit._Cache[name]
            return
        if data_dir is None:
            data_dir = get_dataset_path('svhn_data')
        assert name in ['train', 'test', 'extra'], name
        filename = os.path.join(data_dir, name + '_32x32.mat')
        assert os.path.isfile(filename), \
                "File {} not found! Please download it from {}.".format(filename, SVHN_URL)
        logger.info("Loading {} ...".format(filename))
        data = scipy.io.loadmat(filename)
        self.X = data['X'].transpose(3,0,1,2)
        self.Y = data['y'].reshape((-1))
        self.Y[self.Y==10] = 0
        SVHNDigit._Cache[name] = (self.X, self.Y)

    def size(self):
        return self.X.shape[0]

    def get_data(self):
        n = self.X.shape[0]
        idxs = np.arange(n)
        if self.shuffle:
            self.rng.shuffle(idxs)
        for k in idxs:
            yield [self.X[k], self.Y[k]]

    @staticmethod
    def get_per_pixel_mean():
        """
        return 32x32x3 image
        """
        a = SVHNDigit('train')
        b = SVHNDigit('test')
        c = SVHNDigit('extra')
        return np.concatenate((a.X, b.X, c.X)).mean(axis=0)

if __name__ == '__main__':
    a = SVHNDigit('train')
    b = SVHNDigit.get_per_pixel_mean()
