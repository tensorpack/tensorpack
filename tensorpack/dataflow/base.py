# -*- coding: utf-8 -*-
# File: base.py


import threading
from abc import abstractmethod, ABCMeta
import six
from ..utils.utils import get_rng

__all__ = ['DataFlow', 'ProxyDataFlow', 'RNGDataFlow', 'DataFlowTerminated']


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


# NOTE: we cannot use six here
class DataFlowMeta(ABCMeta):
    """
    DataFlow uses "__iter__()" and "__len__()" instead of
    "get_data()" and "size()". This add back-compatibility.
    """
    def __new__(mcls, name, bases, namespace, **kwargs):

        def hot_patch(required, existing):
            if required not in namespace and existing in namespace:
                namespace[required] = namespace[existing]

        hot_patch('__iter__', 'get_data')
        hot_patch('__len__', 'size')

        return ABCMeta.__new__(mcls, name, bases, namespace, **kwargs)


@six.add_metaclass(DataFlowMeta)
class DataFlow(object):
    """ Base class for all DataFlow """

    @abstractmethod
    def __iter__(self):
        """
        The method to generate datapoints.

        Yields:
            list: The datapoint, i.e. list of components.
        """

    def get_data(self):
        return self.__iter__()

    def __len__(self):
        """
        Returns:
            int: size of this data flow.

        Raises:
            :class:`NotImplementedError` if this DataFlow doesn't have a size.
        """
        raise NotImplementedError()

    def size(self):
        return self.__len__()

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


class RNGDataFlow(DataFlow):
    """ A DataFlow with RNG"""

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

    def __len__(self):
        return self.ds.__len__()

    def __iter__(self):
        return self.ds.__iter__()
