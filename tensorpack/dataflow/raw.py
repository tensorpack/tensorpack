#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: raw.py


import numpy as np
import copy
import six
from six.moves import range
from .base import DataFlow, RNGDataFlow
from ..utils.develop import log_deprecated

__all__ = ['FakeData', 'DataFromQueue', 'DataFromList', 'DataFromGenerator', 'DataFromIterable']


class FakeData(RNGDataFlow):
    """ Generate fake data of given shapes"""

    def __init__(self, shapes, size=1000, random=True, dtype='float32', domain=(0, 1)):
        """
        Args:
            shapes (list): a list of lists/tuples. Shapes of each component.
            size (int): size of this DataFlow.
            random (bool): whether to randomly generate data every iteration.
                Note that merely generating the data could sometimes be time-consuming!
            dtype (str or list): data type as string, or a list of data types.
            domain (tuple or list): (min, max) tuple, or a list of such tuples
        """
        super(FakeData, self).__init__()
        self.shapes = shapes
        self._size = int(size)
        self.random = random
        self.dtype = [dtype] * len(shapes) if isinstance(dtype, six.string_types) else dtype
        self.domain = [domain] * len(shapes) if isinstance(domain, tuple) else domain
        assert len(self.dtype) == len(self.shapes)
        assert len(self.domain) == len(self.domain)

    def size(self):
        return self._size

    def get_data(self):
        if self.random:
            for _ in range(self._size):
                val = []
                for k in range(len(self.shapes)):
                    v = self.rng.rand(*self.shapes[k]) * (self.domain[k][1] - self.domain[k][0]) + self.domain[k][0]
                    val.append(v.astype(self.dtype[k]))
                yield val
        else:
            val = []
            for k in range(len(self.shapes)):
                v = self.rng.rand(*self.shapes[k]) * (self.domain[k][1] - self.domain[k][0]) + self.domain[k][0]
                val.append(v.astype(self.dtype[k]))
            for _ in range(self._size):
                yield copy.copy(val)


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
    """ Wrap a list of datapoints to a DataFlow"""

    def __init__(self, lst, shuffle=True):
        """
        Args:
            lst (list): input list. Each element is a datapoint.
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


class DataFromGenerator(DataFlow):
    """
    Wrap a generator to a DataFlow.
    """
    def __init__(self, gen, size=None):
        """
        Args:
            gen: iterable, or a callable that returns an iterable
            size: deprecated
        """
        if not callable(gen):
            self._gen = lambda: gen
        else:
            self._gen = gen
        if size is not None:
            log_deprecated("DataFromGenerator(size=)", "It doesn't make much sense.", "2018-03-31")

    def get_data(self):
        # yield from
        for dp in self._gen():
            yield dp


class DataFromIterable(DataFlow):
    """ Wrap an iterable of datapoitns to a DataFlow"""
    def __init__(self, iterable):
        """
        Args:
            iterable: an iterable object with length
        """
        self._itr = iterable
        self._len = len(iterable)

    def size(self):
        return self._len

    def get_data(self):
        for dp in self._itr:
            yield dp
