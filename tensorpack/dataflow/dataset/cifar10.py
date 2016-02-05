#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: cifar10.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>
import os, sys
import cPickle
import numpy
from six.moves import urllib
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
    for fname in filenames:
        fo = open(fname, 'rb')
        dic = cPickle.load(fo)
        data = dic['data']
        label = dic['labels']
        fo.close()
        for k in xrange(10000):
            img = data[k].reshape(3, 32, 32)
            img = numpy.transpose(img, [1, 2, 0])
            yield [img, label[k]]

class Cifar10(DataFlow):
    def __init__(self, train_or_test, dir=None):
        """
        Args:
            train_or_test: string either 'train' or 'test'
        """
        assert train_or_test in ['train', 'test']
        if dir is None:
            dir = os.path.join(os.path.dirname(__file__), 'cifar10_data')
        maybe_download_and_extract(dir)

        if train_or_test == 'train':
            self.fs = [os.path.join(
                dir, 'cifar-10-batches-py', 'data_batch_%d' % i) for i in xrange(1, 6)]
        else:
            self.fs = [os.path.join(dir, 'cifar-10-batches-py', 'test_batch')]
        for f in self.fs:
            if not os.path.isfile(f):
                raise ValueError('Failed to find file: ' + f)
        self.train_or_test = train_or_test

    def size(self):
        return 50000 if self.train_or_test == 'train' else 10000

    def get_data(self):
        for k in read_cifar10(self.fs):
            yield k

if __name__ == '__main__':
    ds = Cifar10('train')
    from dataflow.dftools import dump_dataset_images
    dump_dataset_images(ds, '/tmp/cifar', 100)
    #for (img, label) in ds.get_data():
        #from IPython import embed; embed()
        #break

