# -*- coding: utf-8 -*-
# File: svhn.py


import numpy as np
import os

from ...utils import logger
from ...utils.fs import download, get_dataset_path
from ..base import RNGDataFlow

__all__ = ['SVHNDigit']

SVHN_URL = "http://ufldl.stanford.edu/housenumbers/"


class SVHNDigit(RNGDataFlow):
    """
    `SVHN <http://ufldl.stanford.edu/housenumbers/>`_ Cropped Digit Dataset.
    Produces [img, label], img of 32x32x3 in range [0,255], label of 0-9
    """
    _Cache = {}

    def __init__(self, name, data_dir=None, shuffle=True):
        """
        Args:
            name (str): 'train', 'test', or 'extra'.
            data_dir (str): a directory containing the original {train,test,extra}_32x32.mat.
            shuffle (bool): shuffle the dataset.
        """
        self.shuffle = shuffle

        if name in SVHNDigit._Cache:
            self.X, self.Y = SVHNDigit._Cache[name]
            return
        if data_dir is None:
            data_dir = get_dataset_path('svhn_data')
        assert name in ['train', 'test', 'extra'], name
        filename = os.path.join(data_dir, name + '_32x32.mat')
        if not os.path.isfile(filename):
            url = SVHN_URL + os.path.basename(filename)
            logger.info("File {} not found!".format(filename))
            logger.info("Downloading from {} ...".format(url))
            download(url, os.path.dirname(filename))
        logger.info("Loading {} ...".format(filename))
        data = scipy.io.loadmat(filename)
        self.X = data['X'].transpose(3, 0, 1, 2)
        self.Y = data['y'].reshape((-1))
        self.Y[self.Y == 10] = 0
        SVHNDigit._Cache[name] = (self.X, self.Y)

    def __len__(self):
        return self.X.shape[0]

    def __iter__(self):
        n = self.X.shape[0]
        idxs = np.arange(n)
        if self.shuffle:
            self.rng.shuffle(idxs)
        for k in idxs:
            # since svhn is quite small, just do it for safety
            yield [self.X[k], self.Y[k]]

    @staticmethod
    def get_per_pixel_mean(names=('train', 'test', 'extra')):
        """
        Args:
            names (tuple[str]): names of the dataset split

        Returns:
            a 32x32x3 image, the mean of all images in the given datasets
        """
        for name in names:
            assert name in ['train', 'test', 'extra'], name
        images = [SVHNDigit(x).X for x in names]
        return np.concatenate(tuple(images)).mean(axis=0)


try:
    import scipy.io
except ImportError:
    from ...utils.develop import create_dummy_class
    SVHNDigit = create_dummy_class('SVHNDigit', 'scipy.io')  # noqa

if __name__ == '__main__':
    a = SVHNDigit('train')
    b = SVHNDigit.get_per_pixel_mean()
