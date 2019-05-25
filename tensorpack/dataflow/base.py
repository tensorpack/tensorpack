# -*- coding: utf-8 -*-
# File: base.py


import threading
from abc import ABCMeta, abstractmethod
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
        * A dataflow is an iterable. The :meth:`__iter__` method should yield a list or dict each time.
          Note that dict is **partially** supported at the moment: certain dataflow does not support dict.

        * The :meth:`__iter__` method can be either finite (will stop iteration) or infinite
          (will not stop iteration). For a finite dataflow, :meth:`__iter__` can be called
          again immediately after the previous call returned.

        * For many dataflow, the :meth:`__iter__` method is non-reentrant, which means for an dataflow
          instance ``df``, :meth:`df.__iter__` cannot be called before the previous
          :meth:`df.__iter__` call has finished (iteration has stopped).
          When a dataflow is non-reentrant, :meth:`df.__iter__` should throw an exception if
          called before the previous call has finished.
          For such non-reentrant dataflows, if you need to use the same dataflow in two places,
          you need to create two dataflow instances.

        Yields:
            list/dict: The datapoint, i.e. list/dict of components.
        """

    def __len__(self):
        """
        * A dataflow can optionally implement :meth:`__len__`. If not implemented, it will
          throw :class:`NotImplementedError`.

        * It returns an integer representing the size of the dataflow.
          The return value **may not be accurate or meaningful** at all.
          When saying the length is "accurate", it means that
          :meth:`__iter__` will always yield this many of datapoints before it stops iteration.

        * There could be many reasons why :meth:`__len__` is inaccurate.
          For example, some dataflow has dynamic size, if it throws away datapoints on the fly.
          Some dataflow mixes the datapoints between consecutive passes over
          the dataset, due to parallelism and buffering.
          In this case it does not make sense to stop the iteration anywhere.

        * Due to the above reasons, the length is only a rough guidance.
          And it's up to the user how to interpret it.
          Inside tensorpack it's only used in these places:

          + A default ``steps_per_epoch`` in training, but you probably want to customize
            it yourself, especially when using data-parallel trainer.
          + The length of progress bar when processing a dataflow.
          + Used by :class:`InferenceRunner` to get the number of iterations in inference.
            In this case users are **responsible** for making sure that :meth:`__len__` is "accurate".
            This is to guarantee that inference is run on a fixed set of images.

        Returns:
            int: rough size of this dataflow.

        Raises:
            :class:`NotImplementedError` if this DataFlow doesn't have a size.
        """
        raise NotImplementedError()

    def reset_state(self):
        """
        * The caller must guarantee that :meth:`reset_state` should be called **once and only once**
          by the **process that uses the dataflow** before :meth:`__iter__` is called.
          The caller thread of this method should stay alive to keep this dataflow alive.

        * It is meant for certain initialization that involves processes,
          e.g., initialize random number generators (RNG), create worker processes.

          Because it's very common to use RNG in data processing,
          developers of dataflow can also subclass :class:`RNGDataFlow` to have easier access to
          a properly-initialized RNG.

        * A dataflow is not fork-safe after :meth:`reset_state` is called (because this will violate the guarantee).
          There are a few other dataflows that are not fork-safe anytime, which will be mentioned in the docs.

        * You should take the responsibility and follow the above guarantee if you're the caller of a dataflow yourself
          (either when you're using dataflow outside of tensorpack, or if you're writing a wrapper dataflow).

        * Tensorpack's built-in forking dataflows (:class:`MultiProcessRunner`, :class:`MultiProcessMapData`, etc)
          and other component that uses dataflows (:class:`InputSource`)
          already take care of the responsibility of calling this method.
        """
        pass

    # These are the old (overly verbose) names for the methods:
    def get_data(self):
        return self.__iter__()

    def size(self):
        return self.__len__()


class RNGDataFlow(DataFlow):
    """ A DataFlow with RNG"""

    rng = None
    """
    ``self.rng`` is a ``np.random.RandomState`` instance that is initialized
    correctly (with different seeds in each process) in ``RNGDataFlow.reset_state()``.
    """

    def reset_state(self):
        """ Reset the RNG """
        self.rng = get_rng(self)


class ProxyDataFlow(DataFlow):
    """ Base class for DataFlow that proxies another.
        Every method is proxied to ``self.ds`` unless overriden by a subclass.
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
