# -*- coding: utf-8 -*-
# File: base.py


import threading
from abc import abstractmethod, ABCMeta
import six
from ..utils.utils import get_rng

__all__ = ['DataFlow', 'ProxyDataFlow', 'RNGDataFlow', 'DataFlowTerminated',
           'RNGDataFlowSequence', 'ProxyDataFlowSequence', 'DataFlowSequenceSlicer']


class DataFlowTerminated(BaseException):
    """
    An exception indicating that the DataFlow is unable to produce any more
    data, i.e. something wrong happened so that calling :meth:`get_data`
    cannot give a valid iterator any more.
    In most DataFlow this will never be raised.
    """
    pass


class DataFlowReentrantGuard(object):
    """
    A tool to enforce non-reentrancy.
    Mostly used on DataFlow whose :meth:`get_data` is stateful,
    so that multiple instances of the iterator cannot co-exist.
    """
    def __init__(self):
        self._lock = threading.Lock()

    def __enter__(self):
        self._succ = self._lock.acquire(False)
        if not self._succ:
            raise threading.ThreadError("This DataFlow is not reentrant!")

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._lock.release()
        return False


@six.add_metaclass(ABCMeta)
class DataFlow(object):
    """ Base class for all DataFlow """

    @abstractmethod
    def get_data(self):
        """
        The method to generate datapoints.

        Yields:
            list: The datapoint, i.e. list of components.
        """

    def size(self):
        """
        Returns:
            int: size of this data flow.

        Raises:
            :class:`NotImplementedError` if this DataFlow doesn't have a size.
        """
        raise NotImplementedError()

    def reset_state(self):
        """
        Reset state of the dataflow.
        It **has to** be called once and only once before producing datapoints.

        Note:
            1. If the dataflow is forked, each process will call this method
               before producing datapoints.
            2. The caller thread of this method must remain alive to keep this dataflow alive.

        For example, RNG **has to** be reset if used in the DataFlow,
        otherwise it won't work well with prefetching, because different
        processes will have the same RNG state.
        """
        pass

class DataFlowSequence(DataFlow):
    """
    A DataFlow based on a sequence. Can be sliced then.
    Should be similar in look to https://github.com/keras-team/keras/blob/master/keras/utils/data_utils.py class Sequence
    """

    def get_data(self):
        """Create a generator that iterate over the Sequence.
        For infinite one, use RepeatedDataSequence"""
        for item in (self[i] for i in range(len(self))):
            yield item

    @abstractmethod
    def __getitem__(self, index):
        """Gets batch at position `index`.
        # Arguments
            index: position of the batch in the Sequence.
        # Returns
            A data item.
        """
        raise NotImplementedError

    def __len__(self):
        return self.size()


class RNGDataFlow(DataFlow):
    """ A DataFlow with RNG"""

    def reset_state(self):
        """ Reset the RNG """
        self.rng = get_rng(self)


class RNGDataFlowSequence(DataFlowSequence):
    """
    A DataFlowSequence with RNG
    """
    def reset_state(self):
        """ Reset the RNG """
        self.rng = get_rng(self)


class ProxyDataFlow(DataFlow):
    """ Base class for DataFlow that proxies another.
        Every method is proxied to ``self.ds`` unless override by subclass.
    """

    def __init__(self, ds):
        """
        Args:
            ds (DataFlow): DataFlow to proxy.
        """
        self.ds = ds

    def reset_state(self):
        self.ds.reset_state()

    def size(self):
        return self.ds.size()

    def get_data(self):
        return self.ds.get_data()


class ProxyDataFlowSequence(DataFlowSequence):
    """
    A DataFlowSequence proxy.
    """

    def __init__(self, ds):
        """
        Args:
            ds (DataFlow): DataFlow to proxy.
        """
        self.ds = ds

    def reset_state(self):
        self.ds.reset_state()

    def size(self):
        return self.ds.size()

    def __getitem__(self, index):
        return self.ds[index]

class DataFlowSequenceSlicer(ProxyDataFlowSequence):
    """
    A DataFlowSequence proxy, that can slice also if the provided input is slicable.

    how to use it:
    when we need a slice of ds, lets proxy it like this:
    DataFlowSequenceSlicer(ds, i, n)

    """

    def __init__(self, ds, slicing_i = None, slicing_n = None):
        ProxyDataFlowSequence.__init__(self,ds)
        self.slicing_i = slicing_i
        self.slicing_n = slicing_n

    def __getitem__(self, index):
        if None in [self.slicing_n, self.slicing_i]:
            return self.ds[index]
        else:
            return self.ds[index*self.slicing_n + self.slicing_i]

    def size(self):
        if None in [self.slicing_n, self.slicing_i]:
            return self.ds.size()
        else:
            remainder_add = 1 if (self.ds.size() % self.slicing_n >= self.slicing_i) else 0
            return int(self.ds.size() / self.slicing_n) + remainder_add

def is_sequence(ds):
    return isinstance(ds, DataFlowSequence) or (hasattr(ds, '__getitem__') and hasattr(ds, '__len__'))