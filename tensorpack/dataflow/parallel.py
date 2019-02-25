# -*- coding: utf-8 -*-
# File: parallel.py

import atexit
import errno
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
from ..utils.concurrency import (
    StoppableThread, enable_death_signal, ensure_proc_terminate, start_proc_mask_signal)
from ..utils.serialize import dumps, loads
from .base import DataFlow, DataFlowReentrantGuard, DataFlowTerminated, ProxyDataFlow

__all__ = ['PrefetchData', 'MultiProcessPrefetchData',
           'PrefetchDataZMQ', 'MultiThreadPrefetchData']


def _repeat_iter(get_itr):
    while True:
        for x in get_itr():
            yield x


def _bind_guard(sock, name):
    try:
        sock.bind(name)
    except zmq.ZMQError:
        logger.error(
            "ZMQError in socket.bind('{}'). Perhaps you're \
using pipes on a non-local file system. See documentation of PrefetchDataZMQ \
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


class MultiProcessPrefetchData(ProxyDataFlow):
    """
    Prefetch data from a DataFlow using Python multiprocessing utilities.
    It will fork the process calling :meth:`__init__`, collect datapoints from `ds` in each
    process by a Python :class:`multiprocessing.Queue`.

    Note:
        1. (Data integrity) An iterator cannot run faster automatically -- what's happening is
           that the process will be forked ``nr_proc`` times.
           There will be ``nr_proc`` dataflow running in parallel and **independently**.
           As a result, we have the following guarantee on the dataflow correctness:

           a. When ``nr_proc=1``, this dataflow produces the same data as the
              given dataflow in the same order.
           b. When ``nr_proc>1``, if each sample from the given dataflow is i.i.d.,
              then this dataflow produces the **same distribution** of data as the given dataflow.
              This implies that there will be duplication, reordering, etc.
              You probably only want to use it for training.

              For example, if your original dataflow contains no randomness and produces the same first datapoint,
              then after parallel prefetching, the datapoint will be produced ``nr_proc`` times
              at the beginning.
              Even when your original dataflow is fully shuffled, you still need to be aware of the
              `Birthday Paradox <https://en.wikipedia.org/wiki/Birthday_problem>`_
              and know that you'll likely see duplicates.

           To utilize parallelism with stricter data integrity, you can use the parallel versions of `MapData`.
        2. This has more serialization overhead than :class:`PrefetchDataZMQ` when data is large.
        3. You can nest like this: ``PrefetchDataZMQ(PrefetchData(df, nr_proc=a), nr_proc=b)``.
           A total of ``a`` instances of ``df`` worker processes will be created.
        4. fork happens in `__init__`. `reset_state()` is a no-op. The worker processes won't get called.
        5. This DataFlow does support windows. However, Windows requires more strict picklability on processes,
           which means that some code that's forkable on Linux may not be forkable on Windows. If that happens you'll
           need to re-organize some part of code that's not forkable.
    """

    class _Worker(mp.Process):
        def __init__(self, ds, queue, idx):
            super(MultiProcessPrefetchData._Worker, self).__init__()
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

    def __init__(self, ds, nr_prefetch, nr_proc):
        """
        Args:
            ds (DataFlow): input DataFlow.
            nr_prefetch (int): size of the queue to hold prefetched datapoints.
            nr_proc (int): number of processes to use.
        """
        # https://docs.python.org/3.6/library/multiprocessing.html?highlight=process#the-spawn-and-forkserver-start-methods
        if os.name == 'nt':
            logger.warn("MultiProcessPrefetchData does support Windows. \
However, Windows requires more strict picklability on processes, which may \
lead of failure on some of the code.")
        super(MultiProcessPrefetchData, self).__init__(ds)
        try:
            self._size = len(ds)
        except NotImplementedError:
            self._size = -1
        self.nr_proc = nr_proc
        self.nr_prefetch = nr_prefetch

        if nr_proc > 1:
            logger.info("[MultiProcessPrefetchData] Will fork a dataflow more than one times. "
                        "This assumes the datapoints are i.i.d.")

        self.queue = mp.Queue(self.nr_prefetch)
        self.procs = [MultiProcessPrefetchData._Worker(self.ds, self.queue, idx)
                      for idx in range(self.nr_proc)]
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


PrefetchData = MultiProcessPrefetchData


# TODO renamed to MultiProcessDataFlow{,ZMQ} if separated to a new project
class PrefetchDataZMQ(_MultiProcessZMQDataFlow):
    """
    Prefetch data from a DataFlow using multiple processes, with ZeroMQ for communication.
    It will fork the calling process of :meth:`reset_state()`,
    and collect datapoints from the given dataflow in each process by ZeroMQ IPC pipe.

    Note:
        1. (Data integrity) An iterator cannot run faster automatically -- what's happening is
           that the process will be forked ``nr_proc`` times.
           There will be ``nr_proc`` dataflow running in parallel and **independently**.
           As a result, we have the following guarantee on the dataflow correctness:

           a. When ``nr_proc=1``, this dataflow produces the same data as the
              given dataflow in the same order.
           b. When ``nr_proc>1``, if each sample from the given dataflow is i.i.d.,
              then this dataflow produces the **same distribution** of data as the given dataflow.
              This implies that there will be duplication, reordering, etc.
              You probably only want to use it for training.

              For example, if your original dataflow contains no randomness and produces the same first datapoint,
              then after parallel prefetching, the datapoint will be produced ``nr_proc`` times
              at the beginning.
              Even when your original dataflow is fully shuffled, you still need to be aware of the
              `Birthday Paradox <https://en.wikipedia.org/wiki/Birthday_problem>`_
              and know that you'll likely see duplicates.

           To utilize parallelism with stricter data integrity, you can use the parallel versions of `MapData`.
        2. `reset_state()` of the given dataflow will be called **once and only once** in the worker processes.
        3. The fork of processes happened in this dataflow's `reset_state()` method.
           Please note that forking a TensorFlow GPU session may be unsafe.
           If you're managing this dataflow on your own,
           it's better to fork before creating the session.
        4. (Fork-safety) After the fork has happened, this dataflow becomes not fork-safe.
           i.e., if you fork an already reset instance of this dataflow,
           it won't be usable in the forked process. Therefore, do not nest two `PrefetchDataZMQ`.
        5. (Thread-safety) ZMQ is not thread safe. Therefore, do not call :meth:`get_data` of the same dataflow in
           more than 1 threads.
        6. This dataflow does not support windows. Use `MultiProcessPrefetchData` which works on windows.
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
            super(PrefetchDataZMQ._Worker, self).__init__()
            self.ds = ds
            self.conn_name = conn_name
            self.hwm = hwm
            self.idx = idx

        def run(self):
            enable_death_signal(_warn=self.idx == 0)
            self.ds.reset_state()
            context = zmq.Context()
            socket = context.socket(zmq.PUSH)
            socket.set_hwm(self.hwm)
            socket.connect(self.conn_name)
            try:
                while True:
                    for dp in self.ds:
                        socket.send(dumps(dp), copy=False)
            # sigint could still propagate here, e.g. when nested
            except KeyboardInterrupt:
                pass
            finally:
                socket.close(0)
                context.destroy(0)

    def __init__(self, ds, nr_proc=1, hwm=50):
        """
        Args:
            ds (DataFlow): input DataFlow.
            nr_proc (int): number of processes to use.
            hwm (int): the zmq "high-water mark" (queue size) for both sender and receiver.
        """
        super(PrefetchDataZMQ, self).__init__()

        self.ds = ds
        self.nr_proc = nr_proc
        self._hwm = hwm

        self._guard = DataFlowReentrantGuard()
        if nr_proc > 1:
            logger.info("[PrefetchDataZMQ] Will fork a dataflow more than one times. "
                        "This assumes the datapoints are i.i.d.")
        try:
            self._size = ds.__len__()
        except NotImplementedError:
            self._size = -1

    def _recv(self):
        return loads(self.socket.recv(copy=False))

    def __len__(self):
        return self.ds.__len__()

    def __iter__(self):
        with self._guard, _zmq_catch_error('PrefetchDataZMQ'):
            for k in itertools.count():
                if self._size > 0 and k >= self._size:
                    break
                yield self._recv()

    def reset_state(self):
        super(PrefetchDataZMQ, self).reset_state()
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PULL)
        self.socket.set_hwm(self._hwm)
        pipename = _get_pipe_name('dataflow')
        _bind_guard(self.socket, pipename)

        self._procs = [PrefetchDataZMQ._Worker(self.ds, pipename, self._hwm, idx)
                       for idx in range(self.nr_proc)]
        self._start_processes()


# TODO renamed to MultiThreadDataFlow if separated to a new project
class MultiThreadPrefetchData(DataFlow):
    """
    Create multiple dataflow instances and run them each in one thread.
    Collect outputs with a queue.

    Note:
        1. (Data integrity) An iterator cannot run faster automatically -- what's happening is
           that each thread will create a dataflow iterator.
           There will be ``nr_thread`` dataflow running in parallel and **independently**.
           As a result, we have the following guarantee on the dataflow correctness:

           a. When ``nr_thread=1``, this dataflow produces the same data as the
              given dataflow in the same order.
           b. When ``nr_thread>1``, if each sample from the given dataflow is i.i.d.,
              then this dataflow produces the **same distribution** of data as the given dataflow.
              This implies that there will be duplication, reordering, etc.
              You probably only want to use it for training.

              For example, if your original dataflow contains no randomness and produces the same first datapoint,
              then after parallel prefetching, the datapoint will be produced ``nr_thread`` times
              at the beginning.
              Even when your original dataflow is fully shuffled, you still need to be aware of the
              `Birthday Paradox <https://en.wikipedia.org/wiki/Birthday_problem>`_
              and know that you'll likely see duplicates.

           To utilize parallelism with stricter data integrity, you can use the parallel versions of `MapData`.
    """

    class _Worker(StoppableThread):
        def __init__(self, get_df, queue):
            super(MultiThreadPrefetchData._Worker, self).__init__()
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

    def __init__(self, get_df, nr_prefetch, nr_thread):
        """
        Args:
            get_df ( -> DataFlow): a callable which returns a DataFlow.
                Each thread will call this function to get the DataFlow to use.
                Therefore do not return the same DataFlow for each call.
            nr_prefetch (int): size of the queue
            nr_thread (int): number of threads
        """
        assert nr_thread > 0 and nr_prefetch > 0
        self.nr_thread = nr_thread
        self.queue = queue.Queue(maxsize=nr_prefetch)
        self.threads = [
            MultiThreadPrefetchData._Worker(get_df, self.queue)
            for _ in range(nr_thread)]

    def reset_state(self):
        for th in self.threads:
            th.df.reset_state()
            th.start()

    def __len__(self):
        return self.threads[0].df.__len__()

    def __iter__(self):
        while True:
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


if __name__ == '__main__':
    import time
    from .raw import DataFromGenerator
    from .common import FixedSizeData
    x = DataFromGenerator(itertools.count())
    x = FixedSizeData(x, 100)
    x = PrefetchDataZMQ(x, 2)
    x.reset_state()
    for idx, dp in enumerate(x):
        print(dp)
        time.sleep(0.1)
