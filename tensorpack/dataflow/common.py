# -*- coding: UTF-8 -*-
# File: common.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

from __future__ import division
import copy
import numpy as np
from collections import deque
from six.moves import range, map
from .base import DataFlow, ProxyDataFlow, RNGDataFlow
from ..utils import *

__all__ = ['BatchData', 'FixedSizeData', 'MapData',
           'RepeatedData', 'MapDataComponent', 'RandomChooseData',
           'RandomMixData', 'JoinData', 'ConcatData', 'SelectComponent',
           'LocallyShuffleData']

class BatchData(ProxyDataFlow):
    def __init__(self, ds, batch_size, remainder=False):
        """
        Group data in `ds` into batches.

        :param ds: a DataFlow instance. Its component must be either a scalar or a numpy array
        :param remainder: whether to return the remaining data smaller than a batch_size.
            If set True, will possibly return a data point of a smaller 1st dimension.
            Otherwise, all generated data are guranteed to have the same size.
        """
        super(BatchData, self).__init__(ds)
        if not remainder:
            try:
                s = ds.size()
                assert batch_size <= ds.size()
            except NotImplementedError:
                pass
        self.batch_size = batch_size
        self.remainder = remainder

    def size(self):
        ds_size = self.ds.size()
        div = ds_size // self.batch_size
        rem = ds_size % self.batch_size
        if rem == 0:
            return div
        return div + int(self.remainder)

    def get_data(self):
        """
        :returns: produce batched data by tiling data on an extra 0th dimension.
        """
        holder = []
        for data in self.ds.get_data():
            holder.append(data)
            if len(holder) == self.batch_size:
                yield BatchData._aggregate_batch(holder)
                holder = []
        if self.remainder and len(holder) > 0:
            yield BatchData._aggregate_batch(holder)

    @staticmethod
    def _aggregate_batch(data_holder):
        size = len(data_holder[0])
        result = []
        for k in range(size):
            dt = data_holder[0][k]
            if type(dt) in [int, bool]:
                tp = 'int32'
            elif type(dt) == float:
                tp = 'float32'
            else:
                tp = dt.dtype
            try:
                result.append(
                    np.array([x[k] for x in data_holder], dtype=tp))
            except KeyboardInterrupt:
                raise
            except:
                logger.exception("Cannot batch data. Perhaps they are of inconsistent shape?")
                import IPython as IP;
                IP.embed(config=IP.terminal.ipapp.load_default_config())
        return result

class FixedSizeData(ProxyDataFlow):
    """ Generate data from another DataFlow, but with a fixed epoch size.
        The state of the underlying DataFlow is maintained among each epoch.
    """
    def __init__(self, ds, size):
        """
        :param ds: a :mod:`DataFlow` to produce data
        :param size: a int
        """
        super(FixedSizeData, self).__init__(ds)
        self._size = int(size)
        self.itr = None

    def size(self):
        return self._size

    def get_data(self):
        """
        Produce data from ds, stop at size
        """
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
    """ Take data points from another `DataFlow` and produce them until
        it's exhausted for certain amount of times.
    """
    def __init__(self, ds, nr):
        """
        :param ds: a :mod:`DataFlow` instance.
        :param nr: number of times to repeat ds.
            If nr == -1, repeat ds infinitely many times.
        """
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
            for _ in range(self.nr):
                for dp in self.ds.get_data():
                    yield dp

class MapData(ProxyDataFlow):
    """ Apply map/filter a function on the datapoint"""
    def __init__(self, ds, func):
        """
        :param ds: a :mod:`DataFlow` instance.
        :param func: a function that takes a original datapoint, returns a new
            datapoint. return None to skip this data point.
            Note that if you use filter, ds.size() won't be correct.
        """
        super(MapData, self).__init__(ds)
        self.func = func

    def get_data(self):
        for dp in self.ds.get_data():
            ret = self.func(dp)
            if ret is not None:
                yield ret

class MapDataComponent(ProxyDataFlow):
    """ Apply map/filter on the given index in the datapoint"""
    def __init__(self, ds, func, index=0):
        """
        :param ds: a :mod:`DataFlow` instance.
        :param func: a function that takes a datapoint component dp[index], returns a
            new value of dp[index]. return None to skip this datapoint.
            Note that if you use filter, ds.size() won't be correct.
        """
        super(MapDataComponent, self).__init__(ds)
        self.func = func
        self.index = index

    def get_data(self):
        for dp in self.ds.get_data():
            repl = self.func(dp[self.index])
            if repl is not None:
                dp[self.index] = repl   # NOTE modifying
                yield dp

class RandomChooseData(RNGDataFlow):
    """
    Randomly choose from several DataFlow. Stop producing when any of them is
    exhausted.
    """
    def __init__(self, df_lists):
        """
        :param df_lists: list of dataflow, or list of (dataflow, probability) tuple
        """
        super(RandomChooseData, self).__init__()
        if isinstance(df_lists[0], (tuple, list)):
            assert sum([v[1] for v in df_lists]) == 1.0
            self.df_lists = df_lists
        else:
            prob = 1.0 / len(df_lists)
            self.df_lists = [(k, prob) for k in df_lists]

    def reset_state(self):
        super(RandomChooseData, self).reset_state()
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
                itr = self.rng.choice(itrs, p=probs)
                yield next(itr)
        except StopIteration:
            return

class RandomMixData(RNGDataFlow):
    """
    Randomly choose from several dataflow, and will eventually exhaust all dataflow.  So it's a perfect mix.
    """
    def __init__(self, df_lists):
        """
        :param df_lists: list of dataflow.
            All DataFlow in `df_lists` must have :func:`size()` implemented
        """
        super(RandomMixData, self).__init__()
        self.df_lists = df_lists
        self.sizes = [k.size() for k in self.df_lists]

    def reset_state(self):
        super(RandomMixData, self).reset_state()
        for d in self.df_lists:
            d.reset_state()

    def size(self):
        return sum(self.sizes)

    def get_data(self):
        sums = np.cumsum(self.sizes)
        idxs = np.arange(self.size())
        self.rng.shuffle(idxs)
        idxs = np.array(list(map(
            lambda x: np.searchsorted(sums, x, 'right'), idxs)))
        itrs = [k.get_data() for k in self.df_lists]
        assert idxs.max() == len(itrs) - 1, "{}!={}".format(idxs.max(), len(itrs)-1)
        for k in idxs:
            yield next(itrs[k])

class ConcatData(DataFlow):
    """
    Concatenate several dataflows.
    """
    def __init__(self, df_lists):
        """
        :param df_lists: list of :mod:`DataFlow` instances
        """
        self.df_lists = df_lists

    def reset_state(self):
        for d in self.df_lists:
            d.reset_state()

    def size(self):
        return sum([x.size() for x in self.df_lists])

    def get_data(self):
        for d in self.df_lists:
            for dp in d.get_data():
                yield dp

class JoinData(DataFlow):
    """
    Join the components from each DataFlow.

    .. code-block:: none

        e.g.: df1: [dp1, dp2]
              df2: [dp3, dp4]
              join: [dp1, dp2, dp3, dp4]
    """
    def __init__(self, df_lists):
        """
        :param df_lists: list of :mod:`DataFlow` instances
        """
        self.df_lists = df_lists
        self._size = self.df_lists[0].size()
        for d in self.df_lists:
            assert d.size() == self._size, \
                    "All DataFlow must have the same size! {} != {}".format(d.size(), self._size)

    def reset_state(self):
        for d in self.df_lists:
            d.reset_state()

    def size(self):
        return self._size

    def get_data(self):
        itrs = [k.get_data() for k in self.df_lists]
        try:
            while True:
                dp = []
                for itr in itrs:
                    dp.extend(next(itr))
                yield dp
        except StopIteration:
            pass
        finally:
            for itr in itrs:
                del itr

class LocallyShuffleData(ProxyDataFlow, RNGDataFlow):
    def __init__(self, ds, cache_size):
        ProxyDataFlow.__init__(self, ds)
        self.q = deque(maxlen=cache_size)

    def reset_state(self):
        ProxyDataFlow.reset_state(self)
        RNGDataFlow.reset_state(self)
        self.ds_itr = self.ds.get_data()
        self.current_cnt = 0

    def get_data(self):
        for _ in range(self.q.maxlen - len(self.q)):
            try:
                self.q.append(next(self.ds_itr))
            except StopIteration:
                logger.error("LocallyShuffleData: cache_size is larger than the size of ds!")
        while True:
            self.rng.shuffle(self.q)
            for _ in range(self.q.maxlen):
                yield self.q.popleft()
                try:
                    self.q.append(next(self.ds_itr))
                except StopIteration:
                    # produce the rest and return
                    self.rng.shuffle(self.q)
                    for v in self.q:
                        yield v
                    return



def SelectComponent(ds, idxs):
    """
    :param ds: a :mod:`DataFlow` instance
    :param idxs: a list of datapoint component index of the original dataflow
    """
    return MapData(ds, lambda dp: [dp[i] for i in idxs])

