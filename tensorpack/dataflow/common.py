# -*- coding: utf-8 -*-
# File: common.py

from __future__ import division
import six
import numpy as np
from copy import copy
import pprint
import itertools
from termcolor import colored
from collections import deque, defaultdict
from six.moves import range, map
import tqdm

from .base import DataFlow, ProxyDataFlow, RNGDataFlow, DataFlowReentrantGuard
from ..utils import logger
from ..utils.utils import get_tqdm, get_rng, get_tqdm_kwargs
from ..utils.develop import log_deprecated

__all__ = ['TestDataSpeed', 'PrintData', 'BatchData', 'BatchDataByShape', 'FixedSizeData', 'MapData',
           'MapDataComponent', 'RepeatedData', 'RepeatedDataPoint', 'RandomChooseData',
           'RandomMixData', 'JoinData', 'ConcatData', 'SelectComponent',
           'LocallyShuffleData', 'CacheData']


class TestDataSpeed(ProxyDataFlow):
    """ Test the speed of some DataFlow """
    def __init__(self, ds, size=5000, warmup=0):
        """
        Args:
            ds (DataFlow): the DataFlow to test.
            size (int): number of datapoints to fetch.
            warmup (int): warmup iterations
        """
        super(TestDataSpeed, self).__init__(ds)
        self.test_size = int(size)
        self.warmup = int(warmup)

    def get_data(self):
        """ Will run testing at the beginning, then produce data normally. """
        self.start_test()
        for dp in self.ds.get_data():
            yield dp

    def start_test(self):
        """
        Start testing with a progress bar.
        """
        self.ds.reset_state()
        itr = self.ds.get_data()
        if self.warmup:
            for d in tqdm.trange(self.warmup, **get_tqdm_kwargs()):
                next(itr)
        # add smoothing for speed benchmark
        with get_tqdm(total=self.test_size,
                      leave=True, smoothing=0.2) as pbar:
            for idx, dp in enumerate(itr):
                pbar.update()
                if idx == self.test_size - 1:
                    break

    def start(self):
        """
        Alias of start_test.
        """
        self.start_test()


class BatchData(ProxyDataFlow):
    """
    Stack datapoints into batches.
    It produces datapoints of the same number of components as ``ds``, but
    each component has one new extra dimension of size ``batch_size``.
    The batch can be either a list of original components, or (by default)
    a numpy array of original components.
    """

    def __init__(self, ds, batch_size, remainder=False, use_list=False):
        """
        Args:
            ds (DataFlow): When ``use_list=False``, the components of ``ds``
                must be either scalars or :class:`np.ndarray`, and have to be consistent in shapes.
            batch_size(int): batch size
            remainder (bool): When the remaining datapoints in ``ds`` is not
                enough to form a batch, whether or not to also produce the remaining
                data as a smaller batch.
                If set to False, all produced datapoints are guaranteed to have the same batch size.
                If set to True, `ds.size()` must be accurate.
            use_list (bool): if True, each component will contain a list
                of datapoints instead of an numpy array of an extra dimension.
        """
        super(BatchData, self).__init__(ds)
        if not remainder:
            try:
                assert batch_size <= ds.size()
            except NotImplementedError:
                pass
        self.batch_size = int(batch_size)
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
    def _aggregate_batch(data_holder, use_list=False):
        size = len(data_holder[0])
        result = []
        for k in range(size):
            if use_list:
                result.append(
                    [x[k] for x in data_holder])
            else:
                dt = data_holder[0][k]
                if type(dt) in list(six.integer_types) + [bool]:
                    tp = 'int32'
                elif type(dt) == float:
                    tp = 'float32'
                else:
                    try:
                        tp = dt.dtype
                    except AttributeError:
                        raise TypeError("Unsupported type to batch: {}".format(type(dt)))
                try:
                    result.append(
                        np.asarray([x[k] for x in data_holder], dtype=tp))
                except Exception as e:  # noqa
                    logger.exception("Cannot batch data. Perhaps they are of inconsistent shape?")
                    if isinstance(dt, np.ndarray):
                        s = pprint.pformat([x[k].shape for x in data_holder])
                        logger.error("Shape of all arrays to be batched: " + s)
                    try:
                        # open an ipython shell if possible
                        import IPython as IP; IP.embed()    # noqa
                    except ImportError:
                        pass
        return result


class BatchDataByShape(BatchData):
    """
    Group datapoints of the same shape together to batches.
    It doesn't require input DataFlow to be homogeneous anymore: it can have
    datapoints of different shape, and batches will be formed from those who
    have the same shape.

    Note:
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
        self._guard = DataFlowReentrantGuard()

    def reset_state(self):
        super(BatchDataByShape, self).reset_state()
        self.holder = defaultdict(list)

    def get_data(self):
        with self._guard:
            for dp in self.ds.get_data():
                shp = dp[self.idx].shape
                holder = self.holder[shp]
                holder.append(dp)
                if len(holder) == self.batch_size:
                    yield BatchData._aggregate_batch(holder)
                    del holder[:]


class FixedSizeData(ProxyDataFlow):
    """ Generate data from another DataFlow, but with a fixed total count.
    """
    def __init__(self, ds, size, keep_state=True):
        """
        Args:
            ds (DataFlow): input dataflow
            size (int): size
            keep_state (bool): keep the iterator state of ``ds``
                between calls to :meth:`get_data()`, so that the
                next call will continue the previous iteration over ``ds``,
                instead of reinitializing an iterator.

        Examples:

        .. code-block:: none

            ds produces: 1, 2, 3, 4, 5; 1, 2, 3, 4, 5; ...
            FixedSizeData(ds, 3, True): 1, 2, 3; 4, 5, 1; 2, 3, 4; ...
            FixedSizeData(ds, 3, False): 1, 2, 3; 1, 2, 3; ...
            FixedSizeData(ds, 6, False): 1, 2, 3, 4, 5, 1; 1, 2, 3, 4, 5, 1;...
        """
        super(FixedSizeData, self).__init__(ds)
        self._size = int(size)
        self.itr = None
        self._guard = DataFlowReentrantGuard()
        self._keep = keep_state

    def size(self):
        return self._size

    def reset_state(self):
        super(FixedSizeData, self).reset_state()
        self.itr = self.ds.get_data()

    def get_data(self):
        with self._guard:
            if self.itr is None:
                self.itr = self.ds.get_data()
            cnt = 0
            while True:
                try:
                    dp = next(self.itr)
                except StopIteration:
                    self.itr = self.ds.get_data()
                    dp = next(self.itr)

                cnt += 1
                yield dp
                if cnt == self._size:
                    if not self._keep:
                        self.itr = None
                    return


class MapData(ProxyDataFlow):
    """
    Apply a mapper/filter on the DataFlow.

    Note:
        1. Please make sure func doesn't modify the components
           unless you're certain it's safe.
        2. If you discard some datapoints, ``ds.size()`` will be incorrect.
    """

    def __init__(self, ds, func):
        """
        Args:
            ds (DataFlow): input DataFlow
            func (datapoint -> datapoint | None): takes a datapoint and returns a new
                datapoint. Return None to discard this datapoint.
        """
        super(MapData, self).__init__(ds)
        self.func = func

    def get_data(self):
        for dp in self.ds.get_data():
            ret = self.func(copy(dp))  # shallow copy the list
            if ret is not None:
                yield ret


class MapDataComponent(MapData):
    """
    Apply a mapper/filter on a datapoint component.

    Note:
        1. This dataflow itself doesn't modify the datapoints.
           But please make sure func doesn't modify the components
           unless you're certain it's safe.
        2. If you discard some datapoints, ``ds.size()`` will be incorrect.
    """
    def __init__(self, ds, func, index=0):
        """
        Args:
            ds (DataFlow): input DataFlow.
            func (TYPE -> TYPE|None): takes ``dp[index]``, returns a new value for ``dp[index]``.
                return None to discard this datapoint.
            index (int): index of the component.
        """
        self._index = int(index)
        self._func = func
        super(MapDataComponent, self).__init__(ds, self._mapper)

    def _mapper(self, dp):
        r = self._func(dp[self._index])
        if r is None:
            return None
        dp = list(dp)   # shallow copy to avoid modifying the list
        dp[self._index] = r
        return dp


class RepeatedData(ProxyDataFlow):
    """ Take data points from another DataFlow and produce them until
        it's exhausted for certain amount of times. i.e.:
        dp1, dp2, .... dpn, dp1, dp2, ....dpn
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
            raise NotImplementedError("size() is unavailable for infinite dataflow")
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


class RepeatedDataPoint(ProxyDataFlow):
    """ Take data points from another DataFlow and produce them a
    certain number of times. i.e.:
    dp1, dp1, ..., dp1, dp2, ..., dp2, ...
    """

    def __init__(self, ds, nr):
        """
        Args:
            ds (DataFlow): input DataFlow
            nr (int): number of times to repeat each datapoint.
        """
        self.nr = int(nr)
        assert self.nr >= 1, self.nr
        super(RepeatedDataPoint, self).__init__(ds)

    def size(self):
        return self.ds.size() * self.nr

    def get_data(self):
        for dp in self.ds.get_data():
            for _ in range(self.nr):
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
    Perfectly mix datapoints from several DataFlow using their :meth:`size()`.
    Will stop when all DataFlow exhausted.
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
        # TODO run till exception


class ConcatData(DataFlow):
    """
    Concatenate several DataFlow.
    Produce datapoints from each DataFlow and go to the next when one
    DataFlow is exhausted.
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

        df1 produces: [c1, c2]
        df2 produces: [c3, c4]
        joined: [c1, c2, c3, c4]
    """

    def __init__(self, df_lists):
        """
        Args:
            df_lists (list): a list of DataFlow.
                When these dataflows have different sizes, JoinData will stop when any
                of them is exhausted.
        """
        self.df_lists = df_lists

        try:
            self._size = self.df_lists[0].size()
            for d in self.df_lists:
                assert d.size() == self._size, \
                    "All DataFlow must have the same size! {} != {}".format(d.size(), self._size)
        except Exception:
            logger.info("[JoinData] Size check failed for the list of dataflow to be joined!")

    def reset_state(self):
        for d in self.df_lists:
            d.reset_state()

    def size(self):
        """
        Return the minimum size among all.
        """
        return min([k.size() for k in self.df_lists])

    def get_data(self):
        itrs = [k.get_data() for k in self.df_lists]
        try:
            while True:
                dp = []
                for itr in itrs:
                    dp.extend(next(itr))
                yield dp
        except StopIteration:   # some of them are exhausted
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

        original df produces: [c1, c2, c3]
        idxs: [2,1]
        this df: [c3, c2]
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
        self._guard = DataFlowReentrantGuard()

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
        with self._guard:
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


class CacheData(ProxyDataFlow):
    """
    Cache the first pass of a DataFlow completely in memory,
    and produce from the cache thereafter.

    NOTE: The user should not stop the iterator before it has reached the end.
        Otherwise the cache may be incomplete.
    """
    def __init__(self, ds, shuffle=False):
        """
        Args:
            ds (DataFlow): input DataFlow.
            shuffle (bool): whether to shuffle the datapoints before producing them.
        """
        self.shuffle = shuffle
        self._guard = DataFlowReentrantGuard()
        super(CacheData, self).__init__(ds)

    def reset_state(self):
        super(CacheData, self).reset_state()
        if self.shuffle:
            self.rng = get_rng(self)
        self.buffer = []

    def get_data(self):
        with self._guard:
            if len(self.buffer):
                if self.shuffle:
                    self.rng.shuffle(self.buffer)
                for dp in self.buffer:
                    yield dp
            else:
                for dp in self.ds.get_data():
                    yield dp
                    self.buffer.append(dp)


class PrintData(ProxyDataFlow):
    """
    Behave like an identity mapping, but print shape and range of the first few datapoints.

    Example:
        To enable this debugging output, you should place it somewhere in your dataflow like

        .. code-block:: python

            def get_data():
                ds = SomeDataSource('path/to/lmdb')
                ds = SomeInscrutableMappings(ds)
                ds = PrintData(ds, num=2, max_list=2)
                return ds
            ds = get_data()

        The output looks like:

        .. code-block:: none

            [0110 09:22:21 @common.py:589] DataFlow Info:
            datapoint 0<2 with 4 components consists of
               0: float with value 0.0816501893251
               1: ndarray:int32 of shape (64,) in range [0, 10]
               2: ndarray:float32 of shape (64, 64) in range [-1.2248, 1.2177]
               3: list of len 50
                  0: ndarray:int32 of shape (64, 64) in range [-128, 80]
                  1: ndarray:float32 of shape (64, 64) in range [0.8400, 0.6845]
                  ...
            datapoint 1<2 with 4 components consists of
               0: float with value 5.88252075399
               1: ndarray:int32 of shape (64,) in range [0, 10]
               2: ndarray:float32 of shape (64, 64) with range [-0.9011, 0.8491]
               3: list of len 50
                  0: ndarray:int32 of shape (64, 64) in range [-70, 50]
                  1: ndarray:float32 of shape (64, 64) in range [0.7400, 0.3545]
                  ...
    """

    def __init__(self, ds, num=1, label=None, name=None, max_depth=3, max_list=3):
        """
        Args:
            ds (DataFlow): input DataFlow.
            num (int): number of dataflow points to print.
            name (str, optional): name to identify this DataFlow.
            max_depth (int, optional): stop output when too deep recursion in sub elements
            max_list (int, optional): stop output when too many sub elements
        """
        super(PrintData, self).__init__(ds)
        self.num = num

        if label:
            log_deprecated("PrintData(label, ...", "Use PrintData(name, ...  instead.", "2018-05-01")
            self.name = label
        else:
            self.name = name
        self.cnt = 0
        self.max_depth = max_depth
        self.max_list = max_list

    def _analyze_input_data(self, entry, k, depth=1, max_depth=3, max_list=3):
        """
        Gather useful debug information from a datapoint.

        Args:
            entry: the datapoint component
            k (int): index of this component in current datapoint
            depth (int, optional): recursion depth
            max_depth, max_list: same as in :meth:`__init__`.

        Returns:
            string: debug message
        """

        class _elementInfo(object):
            def __init__(self, el, pos, depth=0, max_list=3):
                self.shape = ""
                self.type = type(el).__name__
                self.dtype = ""
                self.range = ""

                self.sub_elements = []

                self.ident = " " * (depth * 2)
                self.pos = pos

                numpy_scalar_types = list(itertools.chain(*np.sctypes.values()))

                if isinstance(el, (int, float, bool)):
                    self.range = " with value {}".format(el)
                elif type(el) is np.ndarray:
                    self.shape = " of shape {}".format(el.shape)
                    self.dtype = ":{}".format(str(el.dtype))
                    self.range = " in range [{}, {}]".format(el.min(), el.max())
                elif type(el) in numpy_scalar_types:
                    self.range = " with value {}".format(el)
                elif isinstance(el, (list)):
                    self.shape = " of len {}".format(len(el))

                    if depth < max_depth:
                        for k, subel in enumerate(el):
                            if k < max_list:
                                self.sub_elements.append(_elementInfo(subel, k, depth + 1, max_list))
                            else:
                                self.sub_elements.append(" " * ((depth + 1) * 2) + '...')
                                break
                    else:
                        if len(el) > 0:
                            self.sub_elements.append(" " * ((depth + 1) * 2) + ' ...')

            def __str__(self):
                strings = []
                vals = (self.ident, self.pos, self.type, self.dtype, self.shape, self.range)
                strings.append("{}{}: {}{}{}{}".format(*vals))

                for k, el in enumerate(self.sub_elements):
                    strings.append(str(el))
                return "\n".join(strings)

        return str(_elementInfo(entry, k, depth, max_list))

    def _get_msg(self, dp):
        msg = [u"datapoint %i<%i with %i components consists of" % (self.cnt, self.num, len(dp))]
        for k, entry in enumerate(dp):
            msg.append(self._analyze_input_data(entry, k, max_depth=self.max_depth, max_list=self.max_list))
        return u'\n'.join(msg)

    def get_data(self):
        for dp in self.ds.get_data():
            # it is important to place this here! otherwise it mixes the output of multiple PrintData
            if self.cnt == 0:
                label = ' (%s)' % self.name if self.name is not None else ""
                logger.info(colored("DataFlow Info%s:" % label, 'cyan'))

            if self.cnt < self.num:
                print(self._get_msg(dp))
                self.cnt += 1
            yield dp

    def reset_state(self):
        super(PrintData, self).reset_state()
        self.cnt = 0
