# -*- coding: utf-8 -*-
# File: parallel.py

import atexit
import pickle
import errno
import traceback
import itertools
import multiprocessing as mp
import os
import sys
import uuid
import weakref
from contextlib import contextmanager
import zmq
from six.moves import queue, range

from ..utils import logger
from ..utils.develop import log_deprecated
from ..utils.concurrency import (
    StoppableThread, enable_death_signal, ensure_proc_terminate, start_proc_mask_signal)
from ..utils.serialize import dumps_once as dumps, loads_once as loads
from .base import DataFlow, DataFlowReentrantGuard, DataFlowTerminated, ProxyDataFlow

__all__ = ['PrefetchData', 'MultiProcessPrefetchData',
           'MultiProcessRunner', 'MultiProcessRunnerZMQ', 'MultiThreadRunner',
           'PrefetchDataZMQ', 'MultiThreadPrefetchData']


# from https://github.com/pytorch/pytorch/blob/master/torch/utils/data/_utils/__init__.py
class _ExceptionWrapper:
    MAGIC = b"EXC_MAGIC"
    """Wraps an exception plus traceback to communicate across threads"""
    def __init__(self, exc_info):
        # It is important that we don't store exc_info, see
        # NOTE [ Python Traceback Reference Cycle Problem ]
        self.exc_type = exc_info[0]
        self.exc_msg = "".join(traceback.format_exception(*exc_info))

    def pack(self):
        return self.MAGIC + pickle.dumps(self)

    @staticmethod
    def unpack(dp):
        if isinstance(dp, bytes) and dp.startswith(_ExceptionWrapper.MAGIC):
            return pickle.loads(dp[len(_ExceptionWrapper.MAGIC):])


def _repeat_iter(get_itr):
    while True:
        yield from get_itr()


def _bind_guard(sock, name):
    try:
        sock.bind(name)
    except zmq.ZMQError:
        logger.error(
            "ZMQError in socket.bind('{}'). Perhaps you're \
using pipes on a non-local file system. See documentation of MultiProcessRunnerZMQ \
for more information.".format(name))
        raise


def _get_pipe_name(name):
    if sys.platform.startswith('linux'):
        # linux supports abstract sockets: http://api.zeromq.org/4-1:zmq-ipc
        pipename = "ipc://@{}-pipe-{}".format(name, str(uuid.uuid1())[:8])
        pipedir = os.environ.get('TENSORPACK_PIPEDIR', None)
        if pipedir is not None:
            logger.warn("TENSORPACK_PIPEDIR is not used on Linux any more! Abstract sockets will be used.")
    else:
        pipedir = os.environ.get('TENSORPACK_PIPEDIR', None)
        if pipedir is not None:
            logger.info("ZMQ uses TENSORPACK_PIPEDIR={}".format(pipedir))
        else:
            pipedir = '.'
        assert os.path.isdir(pipedir), pipedir
        filename = '{}/{}-pipe-{}'.format(pipedir.rstrip('/'), name, str(uuid.uuid1())[:6])
        assert not os.path.exists(filename), "Pipe {} exists! You may be unlucky.".format(filename)
        pipename = "ipc://{}".format(filename)
    return pipename


def del_weakref(x):
    o = x()
    if o is not None:
        o.__del__()


@contextmanager
def _zmq_catch_error(name):
    try:
        yield
    except zmq.ContextTerminated:
        logger.info("[{}] Context terminated.".format(name))
        raise DataFlowTerminated()
    except zmq.ZMQError as e:
        if e.errno == errno.ENOTSOCK:       # socket closed
            logger.info("[{}] Socket closed.".format(name))
            raise DataFlowTerminated()
        else:
            raise
    except Exception:
        raise


class _MultiProcessZMQDataFlow(DataFlow):
    def __init__(self):
        assert os.name != 'nt', "ZMQ IPC doesn't support windows!"
        self._reset_done = False
        self._procs = []

    def reset_state(self):
        """
        All forked dataflows should only be reset **once and only once** in spawned processes.
        Subclasses should call this method with super.
        """
        assert not self._reset_done, "reset_state() was called twice! This violates the API of DataFlow!"
        self._reset_done = True

        # __del__ not guaranteed to get called at exit
        atexit.register(del_weakref, weakref.ref(self))

    def _start_processes(self):
        start_proc_mask_signal(self._procs)

    def __del__(self):
        try:
            if not self._reset_done:
                return
            if not self.context.closed:
                self.socket.close(0)
                self.context.destroy(0)
            for x in self._procs:
                x.terminate()
                x.join(5)
            print("{} successfully cleaned-up.".format(type(self).__name__))
        except Exception:
            pass


class MultiProcessRunner(ProxyDataFlow):
    """
    Running a DataFlow in >=1 processes using Python multiprocessing utilities.
    It will fork the process that calls :meth:`__init__`, collect datapoints from `ds` in each
    process by a Python :class:`multiprocessing.Queue`.

    Note:
        1. (Data integrity) An iterator cannot run faster automatically -- what's happening is
           that the process will be forked ``num_proc`` times.
           There will be ``num_proc`` dataflow running in parallel and **independently**.
           As a result, we have the following guarantee on the dataflow correctness:

           a. When ``num_proc=1``, this dataflow produces the same data as the
              given dataflow in the same order.
           b. When ``num_proc>1``, if each sample from the given dataflow is i.i.d.,
              then this dataflow produces the **same distribution** of data as the given dataflow.
              This implies that there will be duplication, reordering, etc.
              You probably only want to use it for training.

              For example, if your original dataflow contains no randomness and produces the same first datapoint,
              then after parallel prefetching, the datapoint will be produced ``num_proc`` times
              at the beginning.
              Even when your original dataflow is fully shuffled, you still need to be aware of the
              `Birthday Paradox <https://en.wikipedia.org/wiki/Birthday_problem>`_
              and know that you'll likely see duplicates.

           To utilize parallelism with more strict data integrity, you can use
           the parallel versions of :class:`MapData`: :class:`MultiThreadMapData`, :class:`MultiProcessMapData`.
        2. This has more serialization overhead than :class:`MultiProcessRunnerZMQ` when data is large.
        3. You can nest like this: ``MultiProcessRunnerZMQ(MultiProcessRunner(df, num_proc=a), num_proc=b)``.
           A total of ``a`` instances of ``df`` worker processes will be created.
        4. Fork happens in `__init__`. `reset_state()` is a no-op.
           DataFlow in the worker processes will be reset at the time of fork.
        5. This DataFlow does support windows. However, Windows requires more strict picklability on processes,
           which means that some code that's forkable on Linux may not be forkable on Windows. If that happens you'll
           need to re-organize some part of code that's not forkable.
    """

    class _Worker(mp.Process):
        def __init__(self, ds, queue, idx):
            super(MultiProcessRunner._Worker, self).__init__()
            self.ds = ds
            self.queue = queue
            self.idx = idx

        def run(self):
            enable_death_signal(_warn=self.idx == 0)
            # reset all ds so each process will produce different data
            self.ds.reset_state()
            while True:
                for dp in self.ds:
                    self.queue.put(dp)

    def __init__(self, ds, num_prefetch=None, num_proc=None, nr_prefetch=None, nr_proc=None):
        """
        Args:
            ds (DataFlow): input DataFlow.
            num_prefetch (int): size of the queue to hold prefetched datapoints.
                Required.
            num_proc (int): number of processes to use. Required.
            nr_prefetch, nr_proc: deprecated argument names
        """
        if nr_prefetch is not None:
            log_deprecated("MultiProcessRunner(nr_prefetch)", "Renamed to 'num_prefetch'", "2020-01-01")
            num_prefetch = nr_prefetch
        if nr_proc is not None:
            log_deprecated("MultiProcessRunner(nr_proc)", "Renamed to 'num_proc'", "2020-01-01")
            num_proc = nr_proc
        if num_prefetch is None or num_proc is None:
            raise TypeError("Missing argument num_prefetch or num_proc in MultiProcessRunner!")

        # https://docs.python.org/3.6/library/multiprocessing.html?highlight=process#the-spawn-and-forkserver-start-methods
        if os.name == 'nt':
            logger.warn("MultiProcessRunner does support Windows. \
However, Windows requires more strict picklability on processes, which may \
lead of failure on some of the code.")
        super(MultiProcessRunner, self).__init__(ds)
        try:
            self._size = len(ds)
        except NotImplementedError:
            self._size = -1
        assert num_proc > 0, num_proc
        assert num_prefetch > 0, num_prefetch
        self.num_proc = num_proc
        self.num_prefetch = num_prefetch

        if num_proc > 1:
            logger.info("[MultiProcessRunner] Will fork a dataflow more than one times. "
                        "This assumes the datapoints are i.i.d.")

        self.queue = mp.Queue(self.num_prefetch)
        self.procs = [MultiProcessRunner._Worker(self.ds, self.queue, idx)
                      for idx in range(self.num_proc)]
        ensure_proc_terminate(self.procs)
        start_proc_mask_signal(self.procs)

    def __iter__(self):
        for k in itertools.count():
            if self._size > 0 and k >= self._size:
                break
            dp = self.queue.get()
            yield dp

    def reset_state(self):
        # do nothing. all ds are reset once and only once in spawned processes
        pass


class MultiProcessRunnerZMQ(_MultiProcessZMQDataFlow):
    """
    Run a DataFlow in >=1 processes, with ZeroMQ for communication.
    It will fork the calling process of :meth:`reset_state()`,
    and collect datapoints from the given dataflow in each process by ZeroMQ IPC pipe.
    This is typically faster than :class:`MultiProcessRunner`.

    Note:
        1. (Data integrity) An iterator cannot run faster automatically -- what's happening is
           that the process will be forked ``num_proc`` times.
           There will be ``num_proc`` dataflow running in parallel and **independently**.
           As a result, we have the following guarantee on the dataflow correctness:

           a. When ``num_proc=1``, this dataflow produces the same data as the
              given dataflow in the same order.
           b. When ``num_proc>1``, if each sample from the given dataflow is i.i.d.,
              then this dataflow produces the **same distribution** of data as the given dataflow.
              This implies that there will be duplication, reordering, etc.
              You probably only want to use it for training.

              For example, if your original dataflow contains no randomness and produces the same first datapoint,
              then after parallel prefetching, the datapoint will be produced ``num_proc`` times
              at the beginning.
              Even when your original dataflow is fully shuffled, you still need to be aware of the
              `Birthday Paradox <https://en.wikipedia.org/wiki/Birthday_problem>`_
              and know that you'll likely see duplicates.

           To utilize parallelism with more strict data integrity, you can use
           the parallel versions of :class:`MapData`: :class:`MultiThreadMapData`, :class:`MultiProcessMapData`.
        2. `reset_state()` of the given dataflow will be called **once and only once** in the worker processes.
        3. The fork of processes happened in this dataflow's `reset_state()` method.
           Please note that forking a TensorFlow GPU session may be unsafe.
           If you're managing this dataflow on your own,
           it's better to fork before creating the session.
        4. (Fork-safety) After the fork has happened, this dataflow becomes not fork-safe.
           i.e., if you fork an already reset instance of this dataflow,
           it won't be usable in the forked process. Therefore, do not nest two `MultiProcessRunnerZMQ`.
        5. (Thread-safety) ZMQ is not thread safe. Therefore, do not call :meth:`get_data` of the same dataflow in
           more than 1 threads.
        6. This dataflow does not support windows. Use `MultiProcessRunner` which works on windows.
        7. (For Mac only) A UNIX named pipe will be created in the current directory.
           However, certain non-local filesystem such as NFS/GlusterFS/AFS doesn't always support pipes.
           You can change the directory by ``export TENSORPACK_PIPEDIR=/other/dir``.
           In particular, you can use somewhere under '/tmp' which is usually local.

           Note that some non-local FS may appear to support pipes and code
           may appear to run but crash with bizarre error.
           Also note that ZMQ limits the maximum length of pipe path.
           If you hit the limit, you can set the directory to a softlink
           which points to a local directory.
    """

    class _Worker(mp.Process):
        def __init__(self, ds, conn_name, hwm, idx):
            super(MultiProcessRunnerZMQ._Worker, self).__init__()
            self.ds = ds
            self.conn_name = conn_name
            self.hwm = hwm
            self.idx = idx

        def run(self):
            enable_death_signal(_warn=self.idx == 0)
            self.ds.reset_state()
            itr = _repeat_iter(lambda: self.ds)

            context = zmq.Context()
            socket = context.socket(zmq.PUSH)
            socket.set_hwm(self.hwm)
            socket.connect(self.conn_name)
            try:
                while True:
                    try:
                        dp = next(itr)
                        socket.send(dumps(dp), copy=False)
                    except Exception:
                        dp = _ExceptionWrapper(sys.exc_info()).pack()
                        socket.send(dumps(dp), copy=False)
                        raise
            # sigint could still propagate here, e.g. when nested
            except KeyboardInterrupt:
                pass
            finally:
                socket.close(0)
                context.destroy(0)

    def __init__(self, ds, num_proc=1, hwm=50, nr_proc=None):
        """
        Args:
            ds (DataFlow): input DataFlow.
            num_proc (int): number of processes to use.
            hwm (int): the zmq "high-water mark" (queue size) for both sender and receiver.
            nr_proc: deprecated
        """
        if nr_proc is not None:
            log_deprecated("MultiProcessRunnerZMQ(nr_proc)", "Renamed to 'num_proc'", "2020-01-01")
            num_proc = nr_proc
        super(MultiProcessRunnerZMQ, self).__init__()

        self.ds = ds
        self.num_proc = num_proc
        self._hwm = hwm

        if num_proc > 1:
            logger.info("[MultiProcessRunnerZMQ] Will fork a dataflow more than one times. "
                        "This assumes the datapoints are i.i.d.")
        try:
            self._size = ds.__len__()
        except NotImplementedError:
            self._size = -1

    def _recv(self):
        ret = loads(self.socket.recv(copy=False))
        exc = _ExceptionWrapper.unpack(ret)
        if exc is not None:
            logger.error("Exception '{}' in worker:".format(str(exc.exc_type)))
            raise exc.exc_type(exc.exc_msg)
        return ret

    def __len__(self):
        return self.ds.__len__()

    def __iter__(self):
        with self._guard, _zmq_catch_error('MultiProcessRunnerZMQ'):
            for k in itertools.count():
                if self._size > 0 and k >= self._size:
                    break
                yield self._recv()

    def reset_state(self):
        super(MultiProcessRunnerZMQ, self).reset_state()
        self._guard = DataFlowReentrantGuard()
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PULL)
        self.socket.set_hwm(self._hwm)
        pipename = _get_pipe_name('dataflow')
        _bind_guard(self.socket, pipename)

        self._procs = [MultiProcessRunnerZMQ._Worker(self.ds, pipename, self._hwm, idx)
                       for idx in range(self.num_proc)]
        self._start_processes()


class MultiThreadRunner(DataFlow):
    """
    Create multiple dataflow instances and run them each in one thread.
    Collect outputs from them with a queue.

    Note:
        1. (Data integrity) An iterator cannot run faster automatically -- what's happening is
           that each thread will create a dataflow iterator.
           There will be ``num_thread`` dataflow running in parallel and **independently**.
           As a result, we have the following guarantee on the dataflow correctness:

           a. When ``num_thread=1``, this dataflow produces the same data as the
              given dataflow in the same order.
           b. When ``num_thread>1``, if each sample from the given dataflow is i.i.d.,
              then this dataflow produces the **same distribution** of data as the given dataflow.
              This implies that there will be duplication, reordering, etc.
              You probably only want to use it for training.

              For example, if your original dataflow contains no randomness and produces the same first datapoint,
              then after parallel prefetching, the datapoint will be produced ``num_thread`` times
              at the beginning.
              Even when your original dataflow is fully shuffled, you still need to be aware of the
              `Birthday Paradox <https://en.wikipedia.org/wiki/Birthday_problem>`_
              and know that you'll likely see duplicates.

           To utilize parallelism with more strict data integrity, you can use
           the parallel versions of :class:`MapData`: :class:`MultiThreadMapData`, :class:`MultiProcessMapData`.
    """

    class _Worker(StoppableThread):
        def __init__(self, get_df, queue):
            super(MultiThreadRunner._Worker, self).__init__()
            self.df = get_df()
            assert isinstance(self.df, DataFlow), self.df
            self.queue = queue
            self.daemon = True

        def run(self):
            self.df.reset_state()
            try:
                while True:
                    for dp in self.df:
                        if self.stopped():
                            return
                        self.queue_put_stoppable(self.queue, dp)
            except Exception:
                if self.stopped():
                    pass        # skip duplicated error messages
                else:
                    raise
            finally:
                self.stop()

    def __init__(self, get_df, num_prefetch=None, num_thread=None, nr_prefetch=None, nr_thread=None):
        """
        Args:
            get_df ( -> DataFlow): a callable which returns a DataFlow.
                Each thread will call this function to get the DataFlow to use.
                Therefore do not return the same DataFlow object for each call,
                unless your dataflow is stateless.
            num_prefetch (int): size of the queue
            num_thread (int): number of threads
            nr_prefetch, nr_thread: deprecated names
        """
        if nr_prefetch is not None:
            log_deprecated("MultiThreadRunner(nr_prefetch)", "Renamed to 'num_prefetch'", "2020-01-01")
            num_prefetch = nr_prefetch
        if nr_thread is not None:
            log_deprecated("MultiThreadRunner(nr_thread)", "Renamed to 'num_thread'", "2020-01-01")
            num_thread = nr_thread
        if num_prefetch is None or num_thread is None:
            raise TypeError("Missing argument num_prefetch or num_thread in MultiThreadRunner!")

        assert num_thread > 0, num_thread
        assert num_prefetch > 0, num_prefetch
        self.num_thread = num_thread
        self.queue = queue.Queue(maxsize=num_prefetch)
        self.threads = [
            MultiThreadRunner._Worker(get_df, self.queue)
            for _ in range(num_thread)]

        try:
            self._size = self.__len__()
        except NotImplementedError:
            self._size = -1

    def reset_state(self):
        for th in self.threads:
            th.df.reset_state()
            th.start()

    def __len__(self):
        return self.threads[0].df.__len__()

    def __iter__(self):
        for k in itertools.count():
            if self._size > 0 and k >= self._size:
                break
            yield self.queue.get()

    def __del__(self):
        for p in self.threads:
            if p.is_alive():
                p.stop()
                p.join()


class PlasmaPutData(ProxyDataFlow):
    """
    Put each data point to plasma shared memory object store, and yield the object id instead.

    Experimental.
    """
    def __init__(self, ds, socket="/tmp/plasma"):
        self._socket = socket
        super(PlasmaPutData, self).__init__(ds)

    def reset_state(self):
        super(PlasmaPutData, self).reset_state()
        self.client = plasma.connect(self._socket, "", 0)

    def __iter__(self):
        for dp in self.ds:
            oid = self.client.put(dp)
            yield [oid.binary()]


class PlasmaGetData(ProxyDataFlow):
    """
    Take plasma object id from a DataFlow, and retrieve it from plasma shared
    memory object store.

    Experimental.
    """
    def __init__(self, ds, socket="/tmp/plasma"):
        self._socket = socket
        super(PlasmaGetData, self).__init__(ds)

    def reset_state(self):
        super(PlasmaGetData, self).reset_state()
        self.client = plasma.connect(self._socket, "", 0)

    def __iter__(self):
        for dp in self.ds:
            oid = plasma.ObjectID(dp[0])
            dp = self.client.get(oid)
            yield dp


plasma = None
# These plasma code is only experimental
# try:
#     import pyarrow.plasma as plasma
# except ImportError:
#     from ..utils.develop import create_dummy_class
#     PlasmaPutData = create_dummy_class('PlasmaPutData', 'pyarrow')   # noqa
#     PlasmaGetData = create_dummy_class('PlasmaGetData', 'pyarrow')   # noqa


# The old inappropriate names:
PrefetchData = MultiProcessRunner
MultiProcessPrefetchData = MultiProcessRunner
PrefetchDataZMQ = MultiProcessRunnerZMQ
MultiThreadPrefetchData = MultiThreadRunner

if __name__ == '__main__':
    import time
    from .raw import DataFromGenerator
    from .common import FixedSizeData
    x = DataFromGenerator(itertools.count())
    x = FixedSizeData(x, 100)
    x = MultiProcessRunnerZMQ(x, 2)
    x.reset_state()
    for idx, dp in enumerate(x):
        print(dp)
        time.sleep(0.1)
