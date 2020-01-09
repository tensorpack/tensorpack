# -*- coding: utf-8 -*-
# File: common.py

from __future__ import division
import itertools
import numpy as np
import pprint
from collections import defaultdict, deque
from copy import copy
import six
import tqdm
from termcolor import colored

from ..utils import logger
from ..utils.utils import get_rng, get_tqdm, get_tqdm_kwargs
from ..utils.develop import log_deprecated
from .base import DataFlow, DataFlowReentrantGuard, ProxyDataFlow, RNGDataFlow

try:
    from collections.abc import Mapping
except ImportError:
    from collections import Mapping

__all__ = ['TestDataSpeed', 'PrintData', 'BatchData', 'BatchDataByShape', 'FixedSizeData', 'MapData',
           'MapDataComponent', 'RepeatedData', 'RepeatedDataPoint', 'RandomChooseData',
           'RandomMixData', 'JoinData', 'ConcatData', 'SelectComponent',
           'LocallyShuffleData', 'CacheData']


class TestDataSpeed(ProxyDataFlow):
    """ Test the speed of a DataFlow """
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
        self._reset_called = False

    def reset_state(self):
        self._reset_called = True
        super(TestDataSpeed, self).reset_state()

    def __iter__(self):
        """ Will run testing at the beginning, then produce data normally. """
        self.start()
        yield from self.ds

    def start(self):
        """
        Start testing with a progress bar.
        """
        if not self._reset_called:
            self.ds.reset_state()
        itr = self.ds.__iter__()
        if self.warmup:
            for _ in tqdm.trange(self.warmup, **get_tqdm_kwargs()):
                next(itr)
        # add smoothing for speed benchmark
        with get_tqdm(total=self.test_size,
                      leave=True, smoothing=0.2) as pbar:
            for idx, dp in enumerate(itr):
                pbar.update()
                if idx == self.test_size - 1:
                    break


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
            ds (DataFlow): A dataflow that produces either list or dict.
                When ``use_list=False``, the components of ``ds``
                must be either scalars or :class:`np.ndarray`, and have to be consistent in shapes.
            batch_size(int): batch size
            remainder (bool): When the remaining datapoints in ``ds`` is not
                enough to form a batch, whether or not to also produce the remaining
                data as a smaller batch.
                If set to False, all produced datapoints are guaranteed to have the same batch size.
                If set to True, `len(ds)` must be accurate.
            use_list (bool): if True, each component will contain a list
                of datapoints instead of an numpy array of an extra dimension.
        """
        super(BatchData, self).__init__(ds)
        if not remainder:
            try:
                assert batch_size <= len(ds)
            except NotImplementedError:
                pass
        self.batch_size = int(batch_size)
        assert self.batch_size > 0
        self.remainder = remainder
        self.use_list = use_list

    def __len__(self):
        ds_size = len(self.ds)
        div = ds_size // self.batch_size
        rem = ds_size % self.batch_size
        if rem == 0:
            return div
        return div + int(self.remainder)

    def __iter__(self):
        """
        Yields:
            Batched data by stacking each component on an extra 0th dimension.
        """
        holder = []
        for data in self.ds:
            holder.append(data)
            if len(holder) == self.batch_size:
                yield BatchData.aggregate_batch(holder, self.use_list)
                del holder[:]
        if self.remainder and len(holder) > 0:
            yield BatchData.aggregate_batch(holder, self.use_list)

    @staticmethod
    def _batch_numpy(data_list):
        data = data_list[0]
        if isinstance(data, six.integer_types):
            dtype = 'int32'
        elif type(data) == bool:
            dtype = 'bool'
        elif type(data) == float:
            dtype = 'float32'
        elif isinstance(data, (six.binary_type, six.text_type)):
            dtype = 'str'
        else:
            try:
                dtype = data.dtype
            except AttributeError:
                raise TypeError("Unsupported type to batch: {}".format(type(data)))
        try:
            return np.asarray(data_list, dtype=dtype)
        except Exception as e:  # noqa
            logger.exception("Cannot batch data. Perhaps they are of inconsistent shape?")
            if isinstance(data, np.ndarray):
                s = pprint.pformat([x.shape for x in data_list])
                logger.error("Shape of all arrays to be batched: " + s)
            try:
                # open an ipython shell if possible
                import IPython as IP; IP.embed()    # noqa
            except ImportError:
                pass

    @staticmethod
    def aggregate_batch(data_holder, use_list=False):
        """
        Aggregate a list of datapoints to one batched datapoint.

        Args:
            data_holder (list[dp]): each dp is either a list or a dict.
            use_list (bool): whether to batch data into a list or a numpy array.

        Returns:
            dp: either a list or a dict, depend on the inputs.
                Each item is a batched version of the corresponding inputs.
        """
        first_dp = data_holder[0]
        if isinstance(first_dp, (list, tuple)):
            result = []
            for k in range(len(first_dp)):
                data_list = [x[k] for x in data_holder]
                if use_list:
                    result.append(data_list)
                else:
                    result.append(BatchData._batch_numpy(data_list))
        elif isinstance(first_dp, dict):
            result = {}
            for key in first_dp.keys():
                data_list = [x[key] for x in data_holder]
                if use_list:
                    result[key] = data_list
                else:
                    result[key] = BatchData._batch_numpy(data_list)
        else:
            raise ValueError("Data point has to be list/tuple/dict. Got {}".format(type(first_dp)))
        return result


class BatchDataByShape(BatchData):
    """
    Group datapoints of the same shape together to batches.
    It doesn't require input DataFlow to be homogeneous anymore: it can have
    datapoints of different shape, and batches will be formed from those who
    have the same shape.

    Note:
        It is implemented by a dict{shape -> datapoints}.
        Therefore, datapoints of uncommon shapes may never be enough to form a batch and
        never get generated.
    """
    def __init__(self, ds, batch_size, idx):
        """
        Args:
            ds (DataFlow): input DataFlow. ``dp[idx]`` has to be an :class:`np.ndarray`.
            batch_size (int): batch size
            idx (int): ``dp[idx].shape`` will be used to group datapoints.
                Other components are assumed to be batch-able.
        """
        super(BatchDataByShape, self).__init__(ds, batch_size, remainder=False)
        self.idx = idx

    def reset_state(self):
        super(BatchDataByShape, self).reset_state()
        self.holder = defaultdict(list)
        self._guard = DataFlowReentrantGuard()

    def __iter__(self):
        with self._guard:
            for dp in self.ds:
                shp = dp[self.idx].shape
                holder = self.holder[shp]
                holder.append(dp)
                if len(holder) == self.batch_size:
                    yield BatchData.aggregate_batch(holder)
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
                between calls to :meth:`__iter__()`, so that the
                next call will continue the previous iteration over ``ds``,
                instead of reinitializing an iterator.

        Example:

        .. code-block:: none

            ds produces: 1, 2, 3, 4, 5; 1, 2, 3, 4, 5; ...
            FixedSizeData(ds, 3, True): 1, 2, 3; 4, 5, 1; 2, 3, 4; ...
            FixedSizeData(ds, 3, False): 1, 2, 3; 1, 2, 3; ...
            FixedSizeData(ds, 6, False): 1, 2, 3, 4, 5, 1; 1, 2, 3, 4, 5, 1;...
        """
        super(FixedSizeData, self).__init__(ds)
        self._size = int(size)
        self.itr = None
        self._keep = keep_state

    def __len__(self):
        return self._size

    def reset_state(self):
        super(FixedSizeData, self).reset_state()
        self.itr = self.ds.__iter__()
        self._guard = DataFlowReentrantGuard()

    def __iter__(self):
        with self._guard:
            if self.itr is None:
                self.itr = self.ds.__iter__()
            cnt = 0
            while True:
                try:
                    dp = next(self.itr)
                except StopIteration:
                    self.itr = self.ds.__iter__()
                    dp = next(self.itr)

                cnt += 1
                yield dp
                if cnt == self._size:
                    if not self._keep:
                        self.itr = None
                    return


class MapData(ProxyDataFlow):
    """
    Apply a mapper/filter on the datapoints of a DataFlow.

    Note:
        1. Please make sure func doesn't modify its arguments in place,
           unless you're certain it's safe.
        2. If you discard some datapoints, ``len(MapData(ds))`` will be incorrect.

    Example:

        .. code-block:: none

            ds = Mnist('train')  # each datapoint is [img, label]
            ds = MapData(ds, lambda dp: [dp[0] * 255, dp[1]])
    """

    def __init__(self, ds, func):
        """
        Args:
            ds (DataFlow): input DataFlow
            func (datapoint -> datapoint | None): takes a datapoint and returns a new
                datapoint. Return None to discard/skip this datapoint.
        """
        super(MapData, self).__init__(ds)
        self.func = func

    def __iter__(self):
        for dp in self.ds:
            ret = self.func(copy(dp))  # shallow copy the list
            if ret is not None:
                yield ret


class MapDataComponent(MapData):
    """
    Apply a mapper/filter on a datapoint component.

    Note:
        1. This dataflow itself doesn't modify the datapoints.
           But please make sure func doesn't modify its arguments in place,
           unless you're certain it's safe.
        2. If you discard some datapoints, ``len(MapDataComponent(ds, ..))`` will be incorrect.

    Example:

        .. code-block:: none

            ds = Mnist('train')  # each datapoint is [img, label]
            ds = MapDataComponent(ds, lambda img: img * 255, 0)  # map the 0th component
    """
    def __init__(self, ds, func, index=0):
        """
        Args:
            ds (DataFlow): input DataFlow which produces either list or dict.
            func (TYPE -> TYPE|None): takes ``dp[index]``, returns a new value for ``dp[index]``.
                Return None to discard/skip this datapoint.
            index (int or str): index or key of the component.
        """
        self._index = index
        self._func = func
        super(MapDataComponent, self).__init__(ds, self._mapper)

    def _mapper(self, dp):
        r = self._func(dp[self._index])
        if r is None:
            return None
        dp = copy(dp)   # shallow copy to avoid modifying the datapoint
        if isinstance(dp, tuple):
            dp = list(dp)  # to be able to modify it in the next line
        dp[self._index] = r
        return dp


class RepeatedData(ProxyDataFlow):
    """ Take data points from another DataFlow and produce them until
        it's exhausted for certain amount of times. i.e.:
        dp1, dp2, .... dpn, dp1, dp2, ....dpn
    """

    def __init__(self, ds, num):
        """
        Args:
            ds (DataFlow): input DataFlow
            num (int): number of times to repeat ds.
                Set to -1 to repeat ``ds`` infinite times.
        """
        self.num = num
        super(RepeatedData, self).__init__(ds)

    def __len__(self):
        """
        Raises:
            :class:`ValueError` when num == -1.
        """
        if self.num == -1:
            raise NotImplementedError("__len__() is unavailable for infinite dataflow")
        return len(self.ds) * self.num

    def __iter__(self):
        if self.num == -1:
            while True:
                yield from self.ds
        else:
            for _ in range(self.num):
                yield from self.ds


class RepeatedDataPoint(ProxyDataFlow):
    """ Take data points from another DataFlow and produce them a
    certain number of times. i.e.:
    dp1, dp1, ..., dp1, dp2, ..., dp2, ...
    """

    def __init__(self, ds, num):
        """
        Args:
            ds (DataFlow): input DataFlow
            num (int): number of times to repeat each datapoint.
        """
        self.num = int(num)
        assert self.num >= 1, self.num
        super(RepeatedDataPoint, self).__init__(ds)

    def __len__(self):
        return len(self.ds) * self.num

    def __iter__(self):
        for dp in self.ds:
            for _ in range(self.num):
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
            assert sum(v[1] for v in df_lists) == 1.0
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

    def __iter__(self):
        itrs = [v[0].__iter__() for v in self.df_lists]
        probs = np.array([v[1] for v in self.df_lists])
        try:
            while True:
                itr = self.rng.choice(itrs, p=probs)
                yield next(itr)
        except StopIteration:
            return


class RandomMixData(RNGDataFlow):
    """
    Perfectly mix datapoints from several DataFlow using their
    :meth:`__len__()`. Will stop when all DataFlow exhausted.
    """

    def __init__(self, df_lists):
        """
        Args:
            df_lists (list): a list of DataFlow.
                All DataFlow must implement ``__len__()``.
        """
        super(RandomMixData, self).__init__()
        self.df_lists = df_lists
        self.sizes = [len(k) for k in self.df_lists]

    def reset_state(self):
        super(RandomMixData, self).reset_state()
        for d in self.df_lists:
            d.reset_state()

    def __len__(self):
        return sum(self.sizes)

    def __iter__(self):
        sums = np.cumsum(self.sizes)
        idxs = np.arange(self.__len__())
        self.rng.shuffle(idxs)
        idxs = np.array(list(map(
            lambda x: np.searchsorted(sums, x, 'right'), idxs)))
        itrs = [k.__iter__() for k in self.df_lists]
        assert idxs.max() == len(itrs) - 1, "{}!={}".format(idxs.max(), len(itrs) - 1)
        for k in idxs:
            yield next(itrs[k])
        # TODO run till exception


class ConcatData(DataFlow):
    """
    Concatenate several DataFlow.
    Produce datapoints from each DataFlow and start the next when one
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

    def __len__(self):
        return sum(len(x) for x in self.df_lists)

    def __iter__(self):
        for d in self.df_lists:
            yield from d


class JoinData(DataFlow):
    """
    Join the components from each DataFlow. See below for its behavior.

    Note that you can't join a DataFlow that produces lists with one that produces dicts.

    Example:

    .. code-block:: none

        df1 produces: [c1, c2]
        df2 produces: [c3, c4]
        joined: [c1, c2, c3, c4]

        df1 produces: {"a":c1, "b":c2}
        df2 produces: {"c":c3}
        joined: {"a":c1, "b":c2, "c":c3}
    """

    def __init__(self, df_lists):
        """
        Args:
            df_lists (list): a list of DataFlow.
                When these dataflows have different sizes, JoinData will stop when any
                of them is exhausted.
                The list could contain the same DataFlow instance more than once,
                but note that in that case `__iter__` will then also be called many times.
        """
        self.df_lists = df_lists

        try:
            self._size = len(self.df_lists[0])
            for d in self.df_lists:
                assert len(d) == self._size, \
                    "All DataFlow must have the same size! {} != {}".format(len(d), self._size)
        except Exception:
            logger.info("[JoinData] Size check failed for the list of dataflow to be joined!")

    def reset_state(self):
        for d in set(self.df_lists):
            d.reset_state()

    def __len__(self):
        """
        Return the minimum size among all.
        """
        return min(len(k) for k in self.df_lists)

    def __iter__(self):
        itrs = [k.__iter__() for k in self.df_lists]
        try:
            while True:
                all_dps = [next(itr) for itr in itrs]
                if isinstance(all_dps[0], (list, tuple)):
                    dp = list(itertools.chain(*all_dps))
                else:
                    dp = {}
                    for x in all_dps:
                        dp.update(x)
                yield dp
        except StopIteration:   # some of them are exhausted
            pass


def SelectComponent(ds, idxs):
    """
    Select / reorder components from datapoints.

    Args:
        ds (DataFlow): input DataFlow.
        idxs (list[int] or list[str]): a list of component indices/keys.

    Example:

    .. code-block:: none

        original df produces: [c1, c2, c3]
        idxs: [2,1]
        this df: [c3, c2]
    """
    return MapData(ds, lambda dp: [dp[i] for i in idxs])


class LocallyShuffleData(ProxyDataFlow, RNGDataFlow):
    """ Buffer the datapoints from a given dataflow, and shuffle them before producing them.
        This can be used as an alternative when a complete random shuffle is too expensive
        or impossible for the data source.

        This dataflow has the following behavior:

        1. It takes datapoints from the given dataflow `ds` to an internal buffer of fixed size.
           Each datapoint is duplicated for `num_reuse` times.
        2. Once the buffer is full, this dataflow starts to yield data from the beginning of the buffer,
           and new datapoints will be added to the end of the buffer. This is like a FIFO queue.
        3. The internal buffer is shuffled after every `shuffle_interval` datapoints that come from `ds`.

        To maintain shuffling states, this dataflow is not reentrant.

        Datapoints from one pass of `ds` will get mixed with datapoints from a different pass.
        As a result, the iterator of this dataflow will run indefinitely
        because it does not make sense to stop the iteration anywhere.
    """

    def __init__(self, ds, buffer_size, num_reuse=1, shuffle_interval=None, nr_reuse=None):
        """
        Args:
            ds (DataFlow): input DataFlow.
            buffer_size (int): size of the buffer.
            num_reuse (int): duplicate each datapoints several times into the buffer to improve
                speed, but duplication may hurt your model.
            shuffle_interval (int): shuffle the buffer after this many
                datapoints were produced from the given dataflow. Frequent shuffle on large buffer
                may affect speed, but infrequent shuffle may not provide enough randomness.
                Defaults to buffer_size / 3
            nr_reuse: deprecated name for num_reuse
        """
        if nr_reuse is not None:
            log_deprecated("LocallyShuffleData(nr_reuse=...)", "Renamed to 'num_reuse'.", "2020-01-01")
            num_reuse = nr_reuse
        ProxyDataFlow.__init__(self, ds)
        self.q = deque(maxlen=buffer_size)
        if shuffle_interval is None:
            shuffle_interval = int(buffer_size // 3)
        self.shuffle_interval = shuffle_interval
        self.num_reuse = num_reuse
        self._inf_ds = RepeatedData(ds, -1)

    def reset_state(self):
        self._guard = DataFlowReentrantGuard()
        ProxyDataFlow.reset_state(self)
        RNGDataFlow.reset_state(self)
        self._iter_cnt = 0
        self._inf_iter = iter(self._inf_ds)

    def __len__(self):
        return len(self.ds) * self.num_reuse

    def __iter__(self):
        with self._guard:
            for dp in self._inf_iter:
                self._iter_cnt = (self._iter_cnt + 1) % self.shuffle_interval
                # fill queue
                if self._iter_cnt == 0:
                    self.rng.shuffle(self.q)
                for _ in range(self.num_reuse):
                    if self.q.maxlen == len(self.q):
                        yield self.q.popleft()
                    self.q.append(dp)


class CacheData(ProxyDataFlow):
    """
    Completely cache the first pass of a DataFlow in memory,
    and produce from the cache thereafter.

    NOTE: The user should not stop the iterator before it has reached the end.
        Otherwise the cache may be incomplete.
    """
    def __init__(self, ds, shuffle=False):
        """
        Args:
            ds (DataFlow): input DataFlow.
            shuffle (bool): whether to shuffle the cache before yielding from it.
        """
        self.shuffle = shuffle
        super(CacheData, self).__init__(ds)

    def reset_state(self):
        super(CacheData, self).reset_state()
        self._guard = DataFlowReentrantGuard()
        if self.shuffle:
            self.rng = get_rng(self)
        self.buffer = []

    def __iter__(self):
        with self._guard:
            if len(self.buffer):
                if self.shuffle:
                    self.rng.shuffle(self.buffer)
                yield from self.buffer
            else:
                for dp in self.ds:
                    yield dp
                    self.buffer.append(dp)


class PrintData(ProxyDataFlow):
    """
    Behave like an identity proxy, but print shape and range of the first few datapoints.
    Good for debugging.

    Example:
        Place it somewhere in your dataflow like

        .. code-block:: python

            def create_my_dataflow():
                ds = SomeDataSource('path/to/lmdb')
                ds = SomeInscrutableMappings(ds)
                ds = PrintData(ds, num=2, max_list=2)
                return ds
            ds = create_my_dataflow()
            # other code that uses ds

        When datapoints are taken from the dataflow, it will print outputs like:

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

    def __init__(self, ds, num=1, name=None, max_depth=3, max_list=3):
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
        self.name = name
        self.cnt = 0
        self.max_depth = max_depth
        self.max_list = max_list

    def _analyze_input_data(self, entry, k, depth=1, max_depth=3, max_list=3):
        """
        Gather useful debug information from a datapoint.

        Args:
            entry: the datapoint component, either a list or a dict
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
                elif isinstance(el, (list, tuple)):
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
        msg = [colored(u"datapoint %i/%i with %i components consists of" %
               (self.cnt, self.num, len(dp)), "cyan")]
        is_dict = isinstance(dp, Mapping)
        for k, entry in enumerate(dp):
            if is_dict:
                key, value = entry, dp[entry]
            else:
                key, value = k, entry
            msg.append(self._analyze_input_data(value, key, max_depth=self.max_depth, max_list=self.max_list))
        return u'\n'.join(msg)

    def __iter__(self):
        for dp in self.ds:
            # it is important to place this here! otherwise it mixes the output of multiple PrintData
            if self.cnt == 0:
                label = ' (%s)' % self.name if self.name is not None else ""
                logger.info(colored("Contents of DataFlow%s:" % label, 'cyan'))

            if self.cnt < self.num:
                print(self._get_msg(dp))
                self.cnt += 1
            yield dp

    def reset_state(self):
        super(PrintData, self).reset_state()
        self.cnt = 0
