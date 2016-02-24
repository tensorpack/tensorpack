#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: common.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import numpy as np
import copy
from .base import DataFlow, ProxyDataFlow
from .imgaug import AugmentorList, Image
from ..utils import *

__all__ = ['BatchData', 'FixedSizeData', 'FakeData', 'MapData',
           'MapDataComponent', 'RandomChooseData',
           'AugmentImageComponent']

class BatchData(ProxyDataFlow):
    def __init__(self, ds, batch_size, remainder=False):
        """
        Group data in ds into batches
        ds: a DataFlow instance
        remainder: whether to return the remaining data smaller than a batch_size.
            if set True, will possibly return a data point of a smaller 1st dimension
        """
        super(BatchData, self).__init__(ds)
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

class FixedSizeData(ProxyDataFlow):
    """ generate data from another dataflow, but with a fixed epoch size"""
    def __init__(self, ds, size):
        super(FixedSizeData, self).__init__(ds)
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

class RepeatedData(ProxyDataFlow):
    """ repeat another dataflow for certain times
        if nr == -1, repeat infinitely many times
    """
    def __init__(self, ds, nr):
        self.nr = nr
        super(RepeatedData, self).__init__(ds)

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
        self.rng = get_rng(self)

    def size(self):
        return self._size

    def reset_state(self):
        self.rng = get_rng(self)

    def get_data(self):
        for _ in xrange(self._size):
            yield [self.rng.random_sample(k) for k in self.shapes]

class MapData(ProxyDataFlow):
    """ Map a function to the datapoint"""
    def __init__(self, ds, func):
        super(MapData, self).__init_(ds)
        self.func = func

    def get_data(self):
        for dp in self.ds.get_data():
            yield self.func(dp)

class MapDataComponent(ProxyDataFlow):
    """ Apply a function to the given index in the datapoint"""
    def __init__(self, ds, func, index=0):
        super(MapDataComponent, self).__init__(ds)
        self.func = func
        self.index = index

    def get_data(self):
        for dp in self.ds.get_data():
            dp = copy.deepcopy(dp)  # avoid modifying the original dp
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

    def reset_state(self):
        for d in self.df_lists:
            if isinstance(d, tuple):
                d[0].reset_state()
            else:
                d.reset_state()

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
