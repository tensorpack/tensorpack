#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: mnist.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>


import os
from tensorflow.examples.tutorials.mnist import input_data

__all__ = ['Mnist']

class Mnist(object):
    def __init__(self, train_or_test, dir=None):
        """
        Args:
            train_or_test: string either 'train' or 'test'
        """
        if dir is None:
            dir = os.path.join(os.path.dirname(__file__), 'mnist_data')
        self.dataset = input_data.read_data_sets(dir)
        self.train_or_test = train_or_test

    def get_data(self):
        ds = self.dataset.train if self.train_or_test == 'train' else self.dataset.test
        for k in xrange(ds.num_examples):
            img = ds.images[k].reshape((28, 28))
            label = ds.labels[k]
            yield (img, label)

if __name__ == '__main__':
    ds = Mnist('train')
    for (img, label) in ds.get_data():
        from IPython import embed; embed()

