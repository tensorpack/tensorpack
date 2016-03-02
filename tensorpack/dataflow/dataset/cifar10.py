#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: cifar10.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>
import os, sys
import pickle
import numpy as np
import random
from six.moves import urllib, range
import copy
import tarfile
import logging

from ...utils import logger
from ..base import DataFlow

__all__ = ['Cifar10']


DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'

def maybe_download_and_extract(dest_directory):
    """Download and extract the tarball from Alex's website.
       copied from tensorflow example """
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    if os.path.isdir(os.path.join(dest_directory, 'cifar-10-batches-py')):
        logger.info("Found cifar10 data in {}.".format(dest_directory))
        return
    else:
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' % (filepath,
                    float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()
        filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, reporthook=_progress)
        print()
        statinfo = os.stat(filepath)
        print('Succesfully downloaded', filename, statinfo.st_size, 'bytes.')
        tarfile.open(filepath, 'r:gz').extractall(dest_directory)

def read_cifar10(filenames):
    ret = []
    for fname in filenames:
        fo = open(fname, 'rb')
        dic = pickle.load(fo, encoding='bytes')
        data = dic[b'data']
        label = dic[b'labels']
        fo.close()
        for k in range(10000):
            img = data[k].reshape(3, 32, 32)
            img = np.transpose(img, [1, 2, 0])
            ret.append([img, label[k]])
    return ret

def get_filenames(dir):
    filenames = [os.path.join(
            dir, 'cifar-10-batches-py', 'data_batch_%d' % i) for i in range(1, 6)]
    filenames.append(os.path.join(
        dir, 'cifar-10-batches-py', 'test_batch'))
    return filenames

class Cifar10(DataFlow):
    """
    Return [image, label],
        image is 32x32x3 in the range [0,255]
    """
    def __init__(self, train_or_test, shuffle=True, dir=None):
        """
        Args:
            train_or_test: string either 'train' or 'test'
            shuffle: default to True
        """
        assert train_or_test in ['train', 'test']
        if dir is None:
            dir = os.path.join(os.path.dirname(__file__), 'cifar10_data')
        maybe_download_and_extract(dir)

        fnames = get_filenames(dir)
        if train_or_test == 'train':
            self.fs = fnames[:5]
        else:
            self.fs = [fnames[-1]]
        for f in self.fs:
            if not os.path.isfile(f):
                raise ValueError('Failed to find file: ' + f)
        self.train_or_test = train_or_test
        self.dir = dir
        self.data = read_cifar10(self.fs)
        self.shuffle = shuffle

    def size(self):
        return 50000 if self.train_or_test == 'train' else 10000

    def get_data(self):
        idxs = list(range(len(self.data)))
        if self.shuffle:
            random.shuffle(idxs)
        for k in idxs:
            yield self.data[k]

    def get_per_pixel_mean(self):
        """
        return a mean image of all (train and test) images of size 32x32x3
        """
        fnames = get_filenames(self.dir)
        all_imgs = [x[0] for x in read_cifar10(fnames)]
        arr = np.array(all_imgs, dtype='float32')
        mean = np.mean(arr, axis=0)
        return mean

    def get_per_channel_mean(self):
        """
        return three values as mean of each channel
        """
        mean = self.get_per_pixel_mean()
        return np.mean(mean, axis=(0,1))

if __name__ == '__main__':
    ds = Cifar10('train')
    from tensorpack.dataflow.dftools import dump_dataset_images
    mean = ds.get_per_channel_mean()
    print(mean)
    dump_dataset_images(ds, '/tmp/cifar', 100)

    #for (img, label) in ds.get_data():
        #from IPython import embed; embed()
        #break

