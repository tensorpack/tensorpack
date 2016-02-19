#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: common.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import numpy as np
from .base import DataFlow
from .imgaug import AugmentorList, Image

__all__ = ['BatchData', 'FixedSizeData', 'FakeData', 'MapData',
           'MapDataComponent', 'RandomChooseData',
           'AugmentImageComponent']

class BatchData(DataFlow):
    def __init__(self, ds, batch_size, remainder=False):
        """
        Group data in ds into batches
        ds: a DataFlow instance
        remainder: whether to return the remaining data smaller than a batch_size.
            if set True, will possibly return a data point of a smaller 1st dimension
        """
        self.ds = ds
        if not remainder:
            assert batch_size <= ds.size()
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
            dt = data_holder[0][k]
            if type(dt) in [int, bool, long]:
                tp = 'int32'
            elif type(dt) == float:
                tp = 'float32'
            else:
                tp = dt.dtype
            result.append(
                np.array([x[k] for x in data_holder], dtype=tp))
        return result

class FixedSizeData(DataFlow):
    """ generate data from another dataflow, but with a fixed epoch size"""
    def __init__(self, ds, size):
        self.ds = ds
        self._size = size
        self.itr = None

    def size(self):
        return self._size

    def get_data(self):
        if self.itr is None:
            self.itr = self.ds.get_data()
        cnt = 0
        while True:
            try:
                dp = self.itr.next()
            except StopIteration:
                self.itr = self.ds.get_data()
                dp = self.itr.next()

            cnt += 1
            yield dp
            if cnt == self._size:
                return

class RepeatedData(DataFlow):
    """ repeat another dataflow for certain times
        if nr == -1, repeat infinitely many times
    """
    def __init__(self, ds, nr):
        self.nr = nr
        self.ds = ds

    def size(self):
        if self.nr == -1:
            raise RuntimeError("size() is unavailable for infinite dataflow")
        return self.ds.size() * self.nr

    def get_data(self):
        if self.nr == -1:
            while True:
                for dp in self.ds.get_data():
                    yield dp
        else:
            for _ in xrange(self.nr):
                for dp in self.ds.get_data():
                    yield dp

class FakeData(DataFlow):
    """ Build fake random data of given shapes"""
    def __init__(self, shapes, size):
        """
        shapes: list of list/tuple
        """
        self.shapes = shapes
        self._size = size

    def size(self):
        return self._size

    def get_data(self):
        for _ in xrange(self._size):
            yield [np.random.random(k) for k in self.shapes]

class MapData(DataFlow):
    """ Map a function to the datapoint"""
    def __init__(self, ds, func):
        self.ds = ds
        self.func = func

    def size(self):
        return self.ds.size()

    def get_data(self):
        for dp in self.ds.get_data():
            yield self.func(dp)

class MapDataComponent(DataFlow):
    """ Apply a function to the given index in the datapoint"""
    def __init__(self, ds, func, index=0):
        self.ds = ds
        self.func = func
        self.index = index

    def size(self):
        return self.ds.size()

    def get_data(self):
        for dp in self.ds.get_data():
            dp[self.index] = self.func(dp[self.index])
            yield dp

class RandomChooseData(DataFlow):
    """
    Randomly choose from several dataflow. Stop producing when any of its dataflow stops.
    """
    def __init__(self, df_lists):
        """
        df_lists: list of dataflow, or list of (dataflow, probability) tuple
        """
        if isinstance(df_lists[0], (tuple, list)):
            assert sum([v[1] for v in df_lists]) == 1.0
            self.df_lists = df_lists
        else:
            prob = 1.0 / len(df_lists)
            self.df_lists = [(k, prob) for k in df_lists]

    def get_data(self):
        itrs = [v[0].get_data() for v in self.df_lists]
        probs = np.array([v[1] for v in self.df_lists])
        try:
            while True:
                itr = np.random.choice(itrs, p=probs)
                yield next(itr)
        except StopIteration:
            return

def AugmentImageComponent(ds, augmentors, index=0):
    """
    Augment the image in each data point
    Args:
        ds: a DataFlow dataset instance
        augmentors: a list of ImageAugmentor instance
        index: the index of image in each data point. default to be 0
    """
    # TODO reset rng at the beginning of each get_data
    aug = AugmentorList(augmentors)
    return MapDataComponent(
        ds,
        lambda img: aug.augment(Image(img)).arr,
        index)
