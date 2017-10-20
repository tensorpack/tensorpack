#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# File: cifar.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>
#         Yukun Chen <cykustc@gmail.com>

import os
import pickle
import numpy as np
import six
from six.moves import range

from ...utils import logger
from ...utils.fs import download, get_dataset_path
from ..base import RNGDataFlow

__all__ = ['Cifar10', 'Cifar100']


DATA_URL_CIFAR_10 = 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
DATA_URL_CIFAR_100 = 'http://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz'


def maybe_download_and_extract(dest_directory, cifar_classnum):
    """Download and extract the tarball from Alex's website.
       copied from tensorflow example """
    assert cifar_classnum == 10 or cifar_classnum == 100
    if cifar_classnum == 10:
        cifar_foldername = 'cifar-10-batches-py'
    else:
        cifar_foldername = 'cifar-100-python'
    if os.path.isdir(os.path.join(dest_directory, cifar_foldername)):
        logger.info("Found cifar{} data in {}.".format(cifar_classnum, dest_directory))
        return
    else:
        DATA_URL = DATA_URL_CIFAR_10 if cifar_classnum == 10 else DATA_URL_CIFAR_100
        download(DATA_URL, dest_directory)
        filename = DATA_URL.split('/')[-1]
        filepath = os.path.join(dest_directory, filename)
        import tarfile
        tarfile.open(filepath, 'r:gz').extractall(dest_directory)


def read_cifar(filenames, cifar_classnum):
    assert cifar_classnum == 10 or cifar_classnum == 100
    ret = []
    for fname in filenames:
        fo = open(fname, 'rb')
        if six.PY3:
            dic = pickle.load(fo, encoding='bytes')
        else:
            dic = pickle.load(fo)
        data = dic[b'data']
        if cifar_classnum == 10:
            label = dic[b'labels']
            IMG_NUM = 10000  # cifar10 data are split into blocks of 10000
        elif cifar_classnum == 100:
            label = dic[b'fine_labels']
            IMG_NUM = 50000 if 'train' in fname else 10000
        fo.close()
        for k in range(IMG_NUM):
            img = data[k].reshape(3, 32, 32)
            img = np.transpose(img, [1, 2, 0])
            ret.append([img, label[k]])
    return ret


def get_filenames(dir, cifar_classnum):
    assert cifar_classnum == 10 or cifar_classnum == 100
    if cifar_classnum == 10:
        filenames = [os.path.join(
            dir, 'cifar-10-batches-py', 'data_batch_%d' % i) for i in range(1, 6)]
        filenames.append(os.path.join(
            dir, 'cifar-10-batches-py', 'test_batch'))
    elif cifar_classnum == 100:
        filenames = [os.path.join(dir, 'cifar-100-python', 'train'),
                     os.path.join(dir, 'cifar-100-python', 'test')]
    return filenames


class CifarBase(RNGDataFlow):
    def __init__(self, train_or_test, shuffle=True, dir=None, cifar_classnum=10):
        assert train_or_test in ['train', 'test']
        assert cifar_classnum == 10 or cifar_classnum == 100
        self.cifar_classnum = cifar_classnum
        if dir is None:
            dir = get_dataset_path('cifar{}_data'.format(cifar_classnum))
        maybe_download_and_extract(dir, self.cifar_classnum)
        fnames = get_filenames(dir, cifar_classnum)
        if train_or_test == 'train':
            self.fs = fnames[:-1]
        else:
            self.fs = [fnames[-1]]
        for f in self.fs:
            if not os.path.isfile(f):
                raise ValueError('Failed to find file: ' + f)
        self.train_or_test = train_or_test
        self.data = read_cifar(self.fs, cifar_classnum)
        self.dir = dir
        self.shuffle = shuffle

    def size(self):
        return 50000 if self.train_or_test == 'train' else 10000

    def get_data(self):
        idxs = np.arange(len(self.data))
        if self.shuffle:
            self.rng.shuffle(idxs)
        for k in idxs:
            # since cifar is quite small, just do it for safety
            yield self.data[k]

    def get_per_pixel_mean(self):
        """
        return a mean image of all (train and test) images of size 32x32x3
        """
        fnames = get_filenames(self.dir, self.cifar_classnum)
        all_imgs = [x[0] for x in read_cifar(fnames, self.cifar_classnum)]
        arr = np.array(all_imgs, dtype='float32')
        mean = np.mean(arr, axis=0)
        return mean

    def get_per_channel_mean(self):
        """
        return three values as mean of each channel
        """
        mean = self.get_per_pixel_mean()
        return np.mean(mean, axis=(0, 1))


class Cifar10(CifarBase):
    """
    Produces [image, label] in Cifar10 dataset,
    image is 32x32x3 in the range [0,255].
    label is an int.
    """
    def __init__(self, train_or_test, shuffle=True, dir=None):
        """
        Args:
            train_or_test (str): either 'train' or 'test'.
            shuffle (bool): shuffle the dataset.
        """
        super(Cifar10, self).__init__(train_or_test, shuffle, dir, 10)


class Cifar100(CifarBase):
    """ Similar to Cifar10"""
    def __init__(self, train_or_test, shuffle=True, dir=None):
        super(Cifar100, self).__init__(train_or_test, shuffle, dir, 100)


if __name__ == '__main__':
    ds = Cifar10('train')
    mean = ds.get_per_channel_mean()
    print(mean)

    import cv2
    ds.reset_state()
    for i, dp in enumerate(ds.get_data()):
        if i == 100:
            break
        img = dp[0]
        cv2.imwrite("{:04d}.jpg".format(i), img)
