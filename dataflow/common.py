#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: batch.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import numpy as np
from .base import DataFlow

__all__ = ['BatchData', 'FixedSizeData']

class BatchData(DataFlow):
    def __init__(self, ds, batch_size, remainder=False):
        """
        Group data in ds into batches
        ds: a DataFlow instance
        remainder: whether to return the remaining data smaller than a batch_size.
            if set, might return a data point of a different shape
        """
        self.ds = ds
        self.batch_size = batch_size
        self.remainder = remainder

    def size(self):
        ds_size = self.ds.size()
        div = ds_size / self.batch_size
        rem = ds_size % self.batch_size
        if rem == 0:
            return div
        return div + int(self.remainder)

    def get_data(self):
        holder = []
        for data in self.ds.get_data():
            holder.append(data)
            if len(holder) == self.batch_size:
                yield BatchData.aggregate_batch(holder)
                holder = []
        if self.remainder and len(holder) > 0:
            yield BatchData.aggregate_batch(holder)

    @staticmethod
    def aggregate_batch(data_holder):
        size = len(data_holder[0])
        result = []
        for k in xrange(size):
            result.append(
                np.array([x[k] for x in data_holder],
                         dtype=data_holder[0][k].dtype))
        return tuple(result)

class FixedSizeData(DataFlow):
    def __init__(self, ds, size):
        self.ds = ds
        self._size = size

    def size(self):
        return self._size

    def get_data(self):
        cnt = 0
        while True:
            for dp in self.ds.get_data():
                cnt += 1
                yield dp
                if cnt == self._size:
                    return

