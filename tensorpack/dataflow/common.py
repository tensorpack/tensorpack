# -*- coding: UTF-8 -*-
# File: common.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

from __future__ import division
import numpy as np
from termcolor import colored
from collections import deque, defaultdict
from six.moves import range, map
from .base import DataFlow, ProxyDataFlow, RNGDataFlow
from ..utils import logger, get_tqdm

__all__ = ['TestDataSpeed', 'PrintData', 'BatchData', 'BatchDataByShape', 'FixedSizeData', 'MapData',
           'MapDataComponent', 'RepeatedData', 'RandomChooseData',
           'RandomMixData', 'JoinData', 'ConcatData', 'SelectComponent',
           'LocallyShuffleData']


class TestDataSpeed(ProxyDataFlow):
    """ Test the speed of some DataFlow """
    def __init__(self, ds, size=5000):
        """
        Args:
            ds (DataFlow): the DataFlow to test.
            size (int): number of datapoints to fetch.
        """
        super(TestDataSpeed, self).__init__(ds)
        self.test_size = size

    def get_data(self):
        """ Will start testing at the beginning, then produce data normally. """
        self.start_test()
        for dp in self.ds.get_data():
            yield dp

    def start_test(self):
        """
        Start testing with a progress bar.
        """
        self.ds.reset_state()
        # add smoothing for speed benchmark
        with get_tqdm(total=self.test_size,
                      leave=True, smoothing=0.2) as pbar:
            for idx, dp in enumerate(self.ds.get_data()):
                pbar.update()
                if idx == self.test_size - 1:
                    break


class BatchData(ProxyDataFlow):
    """
    Concat datapoints into batches.
    It produces datapoints of the same number of components as ``ds``, but
    each component has one new extra dimension of size ``batch_size``.
    The new component can be a list of the original datapoints, or an ndarray
    of the original datapoints.
    """

    def __init__(self, ds, batch_size, remainder=False, use_list=False):
        """
        Args:
            ds (DataFlow): Its components must be either scalars or :class:`np.ndarray`.
                Each component has to be of the same shape across datapoints.
            batch_size(int): batch size
            remainder (bool): whether to return the remaining data smaller than a batch_size.
                If set True, it will possibly generates a data point of a smaller batch size.
                Otherwise, all generated data are guranteed to have the same size.
            use_list (bool): if True, it will run faster by producing a list
                of datapoints instead of an ndarray of datapoints, avoiding an
                extra copy.
        """
        super(BatchData, self).__init__(ds)
        if not remainder:
            try:
                assert batch_size <= ds.size()
            except NotImplementedError:
                pass
        self.batch_size = batch_size
        self.remainder = remainder
        self.use_list = use_list

    def size(self):
        ds_size = self.ds.size()
        div = ds_size // self.batch_size
        rem = ds_size % self.batch_size
        if rem == 0:
            return div
        return div + int(self.remainder)

    def get_data(self):
        """
        Yields:
            Batched data by stacking each component on an extra 0th dimension.
        """
        holder = []
        for data in self.ds.get_data():
            holder.append(data)
            if len(holder) == self.batch_size:
                yield BatchData._aggregate_batch(holder, self.use_list)
                del holder[:]
        if self.remainder and len(holder) > 0:
            yield BatchData._aggregate_batch(holder, self.use_list)

    @staticmethod
    def _aggregate_batch(data_holder, use_list):
        size = len(data_holder[0])
        result = []
        for k in range(size):
            if use_list:
                result.append(
                    [x[k] for x in data_holder])
            else:
                dt = data_holder[0][k]
                if type(dt) in [int, bool]:
                    tp = 'int32'
                elif type(dt) == float:
                    tp = 'float32'
                else:
                    tp = dt.dtype
                try:
                    result.append(
                        np.asarray([x[k] for x in data_holder], dtype=tp))
                except KeyboardInterrupt:
                    raise
                except:
                    logger.exception("Cannot batch data. Perhaps they are of inconsistent shape?")
                    import IPython as IP
                    IP.embed(config=IP.terminal.ipapp.load_default_config())
        return result


class BatchDataByShape(BatchData):
    """
    Group datapoints of the same shape together to batches.
    It doesn't require input DataFlow to be homogeneous anymore: it can have
    datapoints of different shape, and batches will be formed from those who
    have the same shape.

    It is implemented by a dict{shape -> datapoints}.
    Datapoints of uncommon shapes may never be enough to form a batch and
    never get generated.
    """
    def __init__(self, ds, batch_size, idx):
        """
        Args:
            ds (DataFlow): input DataFlow. ``dp[idx]`` has to be an :class:`np.ndarray`.
            batch_size (int): batch size
            idx (int): ``dp[idx].shape`` will be used to group datapoints.
                Other components are assumed to have the same shape.
        """
        super(BatchDataByShape, self).__init__(ds, batch_size, remainder=False)
        self.idx = idx

    def reset_state(self):
        super(BatchDataByShape, self).reset_state()
        self.holder = defaultdict(list)

    def get_data(self):
        for dp in self.ds.get_data():
            shp = dp[self.idx].shape
            holder = self.holder[shp]
            holder.append(dp)
            if len(holder) == self.batch_size:
                yield BatchData._aggregate_batch(holder)
                del holder[:]


class FixedSizeData(ProxyDataFlow):
    """ Generate data from another DataFlow, but with a fixed size.
        The state of the underlying DataFlow won't be reset when it's exhausted.
    """
    def __init__(self, ds, size):
        """
        Args:
            ds (DataFlow): input dataflow
            size (int): size
        """
        super(FixedSizeData, self).__init__(ds)
        self._size = int(size)
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


class MapData(ProxyDataFlow):
    """ Apply a mapper/filter on the DataFlow"""

    def __init__(self, ds, func):
        """
        Args:
            ds (DataFlow): input DataFlow
            func (datapoint -> datapoint | None): takes a datapoint and returns a new
                datapoint. Return None to discard this data point.
                Note that if you use the filter feature, ``ds.size()`` will be incorrect.
        """
        super(MapData, self).__init__(ds)
        self.func = func

    def get_data(self):
        for dp in self.ds.get_data():
            ret = self.func(dp)
            if ret is not None:
                yield ret


class MapDataComponent(MapData):
    """ Apply a mapper/filter on a datapoint component"""
    def __init__(self, ds, func, index=0):
        """
        Args:
            ds (DataFlow): input DataFlow.
            func (TYPE -> TYPE|None): takes ``dp[index]``, returns a new value for ``dp[index]``.
                return None to discard this datapoint.
                Note that if you use the filter feature, ``ds.size()`` will be incorrect.
            index (int): index of the component.
        """
        def f(dp):
            r = func(dp[index])
            if r is None:
                return None
            dp[index] = r
            return dp
        super(MapDataComponent, self).__init__(ds, f)


class RepeatedData(ProxyDataFlow):
    """ Take data points from another DataFlow and produce them until
        it's exhausted for certain amount of times.
    """

    def __init__(self, ds, nr):
        """
        Args:
            ds (DataFlow): input DataFlow
            nr (int): number of times to repeat ds.
                Set to -1 to repeat ``ds`` infinite times.
        """
        self.nr = nr
        super(RepeatedData, self).__init__(ds)

    def size(self):
        """
        Raises:
            :class:`ValueError` when nr == -1.
        """
        if self.nr == -1:
            raise ValueError("size() is unavailable for infinite dataflow")
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


class RandomChooseData(RNGDataFlow):
    """
    Randomly choose from several DataFlow.
    Stop producing when any of them is exhausted.
    """

    def __init__(self, df_lists):
        """
        Args:
            df_lists (list): a list of DataFlow, or a list of (DataFlow, probability) tuples.
                Probabilities must sum to 1 if used.
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
    Perfectly mix datapoints from several DataFlow. Will stop when all DataFlow exhausted.
    """

    def __init__(self, df_lists):
        """
        Args:
            df_lists (list): a list of DataFlow.
                All DataFlow must implement ``size()``.
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
        assert idxs.max() == len(itrs) - 1, "{}!={}".format(idxs.max(), len(itrs) - 1)
        for k in idxs:
            yield next(itrs[k])


class ConcatData(DataFlow):
    """
    Concatenate several dataflows. Produce datapoints from them one by one.
    """

    def __init__(self, df_lists):
        """
        Args:
            df_lists (list): a list of DataFlow.
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

    Examples:

    .. code-block:: none

        dp1: [c1, c2]
        dp2: [c3, c4]
        join: [c1, c2, c3, c4]
    """

    def __init__(self, df_lists):
        """
        Args:
            df_lists (list): a list of DataFlow. All must have the same ``size()``.
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


def SelectComponent(ds, idxs):
    """
    Select / reorder components from datapoints.

    Args:
        ds (DataFlow): input DataFlow.
        idxs (list[int]): a list of component indices.

    Example:

    .. code-block:: none

        dp: [c1, c2, c3]
        idxs: [2,1]
        output dp: [c3, c2]
    """
    return MapData(ds, lambda dp: [dp[i] for i in idxs])


class LocallyShuffleData(ProxyDataFlow, RNGDataFlow):
    """ Maintain a pool to buffer datapoints, and shuffle before producing them.
        This can be used as an alternative when a complete random read is too expensive
        or impossible for the data source.
    """

    def __init__(self, ds, buffer_size, nr_reuse=1, shuffle_interval=None):
        """
        Args:
            ds (DataFlow): input DataFlow.
            buffer_size (int): size of the buffer.
            nr_reuse (int): reuse each datapoints several times to improve
                speed, but may hurt your model.
            shuffle_interval (int): shuffle the buffer after this many
                datapoints went through it. Frequent shuffle on large buffer
                may affect speed, but infrequent shuffle may affect
                randomness. Defaults to buffer_size / 3
        """
        ProxyDataFlow.__init__(self, ds)
        self.q = deque(maxlen=buffer_size)
        if shuffle_interval is None:
            shuffle_interval = int(buffer_size // 3)
        self.shuffle_interval = shuffle_interval
        self.nr_reuse = nr_reuse

    def reset_state(self):
        ProxyDataFlow.reset_state(self)
        RNGDataFlow.reset_state(self)
        self.ds_itr = RepeatedData(self.ds, -1).get_data()
        self.current_cnt = 0

    def _add_data(self):
        dp = next(self.ds_itr)
        for _ in range(self.nr_reuse):
            self.q.append(dp)

    def get_data(self):
        # fill queue
        while self.q.maxlen > len(self.q):
            self._add_data()

        sz = self.size()
        cnt = 0
        while True:
            self.rng.shuffle(self.q)
            for _ in range(self.shuffle_interval):
                # the inner loop maintains the queue size (almost) unchanged
                for _ in range(self.nr_reuse):
                    yield self.q.popleft()
                cnt += self.nr_reuse
                if cnt >= sz:
                    return
                self._add_data()


class PrintData(ProxyDataFlow):
    """
    Behave like an identity mapping but print shapes of produced datapoints once during construction.

    Attributes:
        label (str): label to identify the data when using this debugging on multiple places.
        num (int): number of iterations

    Example:
        To enable this debugging output, you should place it somewhere in your dataflow like

        .. code-block:: python

            def get_data():
                ds = CaffeLMDB('path/to/lmdb')
                ds = SomeInscrutableMappings(ds)
                ds = PrintData(ds, num=2)
                return ds
            ds = get_data()

        The output looks like:

        .. code-block:: none

            [0110 09:22:21 @common.py:589] DataFlow Info:
            datapoint 0<2 with 4 components consists of
               dp 0: is float of shape () with range [0.0816501893251]
               dp 1: is ndarray of shape (64, 64) with range [0.1300, 0.6895]
               dp 2: is ndarray of shape (64, 64) with range [-1.2248, 1.2177]
               dp 3: is ndarray of shape (9, 9) with range [-0.6045, 0.6045]
            datapoint 1<2 with 4 components consists of
               dp 0: is float of shape () with range [5.88252075399]
               dp 1: is ndarray of shape (64, 64) with range [0.0072, 0.9371]
               dp 2: is ndarray of shape (64, 64) with range [-0.9011, 0.8491]
               dp 3: is ndarray of shape (9, 9) with range [-0.5585, 0.5585]
    """

    def __init__(self, ds, num=1, label=""):
        """
        Args:
            ds (DataFlow): input DataFlow.
            num (int): number of dataflow points to print.
            label (str, optional): label to identify this call, when using multiple times
        """
        super(PrintData, self).__init__(ds)
        self.num = num
        self.label = label
        self.print_info()

    def _analyze_input_data(self, el, k, depth=1):
        """
        Gather useful debug information from a datapoint.

        Args:
            el: Description
            k (int): position in current datapoint
            depth (int, optional): recursion depth

        Todo:
            * call this recursively and stop when depth>n for some n if an element is a list

        Returns:
            string: debug message
        """
        if isinstance(el, list):
            return "%s is list of %i elements" % (" " * (depth * 2), len(el))
        else:
            el_type = el.__class__.__name__

            if isinstance(el, (int, float, bool)):
                el_max = el_min = el
                el_shape = "()"
                el_range = el
            else:
                el_shape = "n.A."
                if hasattr(el, 'shape'):
                    el_shape = el.shape

                el_max, el_min = None, None
                if hasattr(el, 'max'):
                    el_max = el.max()
                if hasattr(el, 'min'):
                    el_min = el.min()

                el_range = ("None, None")
                if el_max is not None or el_min is not None:
                    el_range = "%.4f, %.4f" % (el_min, el_max)

            return ("%s dp %i: is %s of shape %s with range [%s]" % (" " * (depth * 2), k, el_type, el_shape, el_range))

    def print_info(self):
        """
        Dump gathered debugging information to stdout.
        """
        def cutoff(gen, num=1):
            """
            Stop a generator after n iterations.

            Args:
                gen (PyGenObject): arbitrary generator
                num (int, optional): number of maximal iterations

            Yields:
                element from generator object
            """

            c = 0
            for el in gen:
                yield el
                c += 1
                if c == num:
                    break

        ds = self.ds
        ds.reset_state()

        msg = [""]
        for i, dummy in enumerate(cutoff(ds.get_data(), self.num)):
            if isinstance(dummy, list):
                msg.append("datapoint %i<%i with %i components consists of" % (i, self.num, len(dummy)))
                for k, entry in enumerate(dummy):
                    msg.append(self._analyze_input_data(entry, k))
        label = "" if self.label is "" else " (" + self.label + ")"
        logger.info(colored("DataFlow Info%s:" % label, 'cyan') + '\n'.join(msg))

        self.ds.reset_state()

    def get_data(self):
        return self.ds.get_data()
