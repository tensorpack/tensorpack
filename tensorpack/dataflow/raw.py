#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: raw.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import numpy as np
import copy
from six.moves import range
from .base import DataFlow, RNGDataFlow

__all__ = ['FakeData', 'DataFromQueue', 'DataFromList']


class FakeData(RNGDataFlow):
    """ Generate fake data of given shapes"""

    def __init__(self, shapes, size=1000, random=True, dtype='float32'):
        """
        Args:
            shapes (list): a list of lists/tuples. Shapes of each component.
            size (int): size of this DataFlow.
            random (bool): whether to randomly generate data every iteration.
                Note that merely generating the data could sometimes be time-consuming!
            dtype (str): data type.
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
                yield copy.deepcopy(v)


class DataFromQueue(DataFlow):
    """ Produce data from a queue """
    def __init__(self, queue):
        """
        Args:
            queue (queue): a queue with ``get()`` method.
        """
        self.queue = queue

    def get_data(self):
        while True:
            yield self.queue.get()


class DataFromList(RNGDataFlow):
    """ Produce data from a list"""

    def __init__(self, lst, shuffle=True):
        """
        Args:
            lst (list): input list.
            shuffle (bool): shuffle data.
        """
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
            idxs = np.arange(len(self.lst))
            self.rng.shuffle(idxs)
            for k in idxs:
                yield self.lst[k]
