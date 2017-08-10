#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# File: base.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import threading
from abc import abstractmethod, ABCMeta
import six
from ..utils.utils import get_rng

__all__ = ['DataFlow', 'ProxyDataFlow', 'RNGDataFlow', 'DataFlowTerminated']


class DataFlowTerminated(BaseException):
    """
    An exception indicating that the DataFlow is unable to produce any more data:
    calling :meth:`get_data` will not give a valid iterator any more.
    In most DataFlow this will not be raised.
    """
    pass


class DataFlowReentrantGuard(object):
    """
    A tool to enforce thread-level non-reentrancy on DataFlow.
    """
    def __init__(self):
        self._lock = threading.Lock()

    def __enter__(self):
        self._succ = self._lock.acquire(False)
        if not self._succ:
            raise threading.ThreadError("This DataFlow cannot be reused under different threads!")

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
        Reset state of the dataflow. It has to be called before producing datapoints.

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
    """ Base class for DataFlow that proxies another"""

    def __init__(self, ds):
        """
        Args:
            ds (DataFlow): DataFlow to proxy.
        """
        self.ds = ds

    def reset_state(self):
        """
        Reset state of the proxied DataFlow.
        """
        self.ds.reset_state()

    def size(self):
        return self.ds.size()

    def get_data(self):
        return self.ds.get_data()
