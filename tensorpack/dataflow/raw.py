#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: raw.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import numpy as np
from six.moves import range
from .base import DataFlow, RNGDataFlow

__all__ = ['FakeData', 'DataFromQueue', 'DataFromList']

class FakeData(RNGDataFlow):
    """ Generate fake fixed data of given shapes"""
    def __init__(self, shapes, size, random=True, dtype='float32'):
        """
        :param shapes: a list of lists/tuples
        :param size: size of this DataFlow
        """
        super(FakeData, self).__init__()
        self.shapes = shapes
        self._size = int(size)
        self.random = random
        self.dtype = dtype

    def size(self):
        return self._size

    def get_data(self):
        if self.random:
            for _ in range(self._size):
                yield [self.rng.rand(*k).astype(self.dtype) for k in self.shapes]
        else:
            v = [self.rng.rand(*k).astype(self.dtype) for k in self.shapes]
            for _ in range(self._size):
                yield v

class DataFromQueue(DataFlow):
    """ Produce data from a queue """
    def __init__(self, queue):
        self.queue = queue

    def get_data(self):
        while True:
            yield self.queue.get()


class DataFromList(RNGDataFlow):
    """ Produce data from a list"""
    def __init__(self, lst, shuffle=True):
        super(DataFromList, self).__init__()
        self.lst = lst
        self.shuffle = shuffle

    def size(self):
        return len(self.lst)

    def get_data(self):
        if not self.shuffle:
            for k in self.lst:
                yield k
        else:
            idxs = self.rng.shuffle(np.arange(len(self.lst)))
            for k in idxs:
                yield self.lst[k]

