# -*- coding: utf-8 -*-
# File: parallel.py

import sys
import weakref
from contextlib import contextmanager
import multiprocessing as mp
import itertools
from six.moves import range, zip, queue
import errno
import uuid
import os
import zmq
import atexit

from .base import DataFlow, ProxyDataFlow, DataFlowTerminated, DataFlowReentrantGuard
from ..utils.concurrency import (ensure_proc_terminate,
                                 mask_sigint, start_proc_mask_signal,
                                 enable_death_signal,
                                 StoppableThread)
from ..utils.serialize import loads, dumps
from ..utils import logger
from ..utils.gpu import change_gpu

__all__ = ['PrefetchData', 'MultiProcessPrefetchData',
           'PrefetchDataZMQ', 'PrefetchOnGPUs', 'MultiThreadPrefetchData']


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
        All forked dataflows are reset **once and only once** in spawned processes.
        Nothing more can be done when calling this method.
        """
        if self._reset_done:
            return
        self._reset_done = True

        # __del__ not guranteed to get called at exit
        atexit.register(del_weakref, weakref.ref(self))

        self._reset_once()  # build processes

    def _reset_once(self):
        pass

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
        1. An iterator cannot run faster automatically -- what's happenning is
           that the underlying dataflow will be forked ``nr_proc`` times.
           As a result, we have the following guarantee on the dataflow correctness:

           a. When ``nr_proc=1``, the dataflow produces the same data as ``ds`` in the same order.
           b. When ``nr_proc>1``, the dataflow produces the same distribution
              of data as ``ds`` if each sample from ``ds`` is i.i.d. (e.g. fully shuffled).
              You probably only want to use it for training.
        2. This has more serialization overhead than :class:`PrefetchDataZMQ` when data is large.
        3. You can nest like this: ``PrefetchDataZMQ(PrefetchData(df, nr_proc=a), nr_proc=b)``.
           A total of ``a`` instances of ``df`` worker processes will be created.
        4. fork happens in `__init__`. `reset_state()` is a no-op. The worker processes won't get called.
    """

    class _Worker(mp.Process):
        def __init__(self, ds, queue):
            super(MultiProcessPrefetchData._Worker, self).__init__()
            self.ds = ds
            self.queue = queue

        def run(self):
            enable_death_signal()
            # reset all ds so each process will produce different data
            self.ds.reset_state()
            while True:
                for dp in self.ds.get_data():
                    self.queue.put(dp)

    def __init__(self, ds, nr_prefetch, nr_proc):
        """
        Args:
            ds (DataFlow): input DataFlow.
            nr_prefetch (int): size of the queue to hold prefetched datapoints.
            nr_proc (int): number of processes to use.
        """
        if os.name == 'nt':
            logger.warn("MultiProcessPrefetchData does support windows. \
However, windows requires more strict picklability on processes, which may \
lead of failure on some of the code.")
        super(MultiProcessPrefetchData, self).__init__(ds)
        try:
            self._size = ds.size()
        except NotImplementedError:
            self._size = -1
        self.nr_proc = nr_proc
        self.nr_prefetch = nr_prefetch

        if nr_proc > 1:
            logger.info("[MultiProcessPrefetchData] Will fork a dataflow more than one times. "
                        "This assumes the datapoints are i.i.d.")

        self.queue = mp.Queue(self.nr_prefetch)
        self.procs = [MultiProcessPrefetchData._Worker(self.ds, self.queue)
                      for _ in range(self.nr_proc)]
        ensure_proc_terminate(self.procs)
        start_proc_mask_signal(self.procs)

    def get_data(self):
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
    Prefetch data from a DataFlow using multiple processes, with ZeroMQ for
    communication.
    It will fork the calling process of :meth:`reset_state()`,
    and collect datapoints from the given dataflow in each process by ZeroMQ IPC pipe.

    Note:
        1. An iterator cannot run faster automatically -- what's happenning is
           that the underlying dataflow will be forked ``nr_proc`` times.
           As a result, we have the following guarantee on the dataflow correctness:

           a. When ``nr_proc=1``, this dataflow produces the same data as the
                given dataflow in the same order.
           b. When ``nr_proc>1``, if each sample from the given dataflow is i.i.d. (e.g. fully shuffled),
                then this dataflow produces the **same distribution** of data as the given dataflow.
                This implies that there will be duplication, reordering, etc.
                You probably only want to use it for training.
                If the samples are not i.i.d., the behavior is undefined.
        2. `reset_state()` of the given dataflow will be called **once and only once** in the worker processes.
        3. The fork of processes happened in this dataflow's `reset_state()` method.
           Please note that forking a TensorFlow GPU session may be unsafe.
           If you're managing this dataflow on your own,
           it's better to fork before creating the session.
        4. After the fork has happened, this dataflow becomes not fork-safe.
           i.e., if you fork an already reset instance of this dataflow,
           it won't be usable in the forked process.
        5. Do not nest two `PrefetchDataZMQ`.
        6. By default, a UNIX named pipe will be created in the current directory.
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
        def __init__(self, ds, conn_name, hwm):
            super(PrefetchDataZMQ._Worker, self).__init__()
            self.ds = ds
            self.conn_name = conn_name
            self.hwm = hwm

        def run(self):
            enable_death_signal()
            self.ds.reset_state()
            context = zmq.Context()
            socket = context.socket(zmq.PUSH)
            socket.set_hwm(self.hwm)
            socket.connect(self.conn_name)
            try:
                while True:
                    for dp in self.ds.get_data():
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
            self._size = ds.size()
        except NotImplementedError:
            self._size = -1

    def _recv(self):
        return loads(self.socket.recv(copy=False).bytes)

    def size(self):
        return self.ds.size()

    def get_data(self):
        with self._guard, _zmq_catch_error('PrefetchDataZMQ'):
            for k in itertools.count():
                if self._size > 0 and k >= self._size:
                    break
                yield self._recv()

    def _reset_once(self):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PULL)
        self.socket.set_hwm(self._hwm)
        pipename = _get_pipe_name('dataflow')
        _bind_guard(self.socket, pipename)

        self._procs = [PrefetchDataZMQ._Worker(self.ds, pipename, self._hwm)
                       for _ in range(self.nr_proc)]
        self._start_processes()


class PrefetchOnGPUs(PrefetchDataZMQ):
    """
    Similar to :class:`PrefetchDataZMQ`,
    but prefetch with each process having its own ``CUDA_VISIBLE_DEVICES`` variable
    mapped to one GPU.
    """

    def __init__(self, ds, gpus):
        """
        Args:
            ds (DataFlow): input DataFlow.
            gpus (list[int]): list of GPUs to use. Will also start this number of processes.
        """
        self.gpus = gpus
        super(PrefetchOnGPUs, self).__init__(ds, len(gpus))

    def _start_processes(self):
        with mask_sigint():
            for gpu, proc in zip(self.gpus, self._procs):
                with change_gpu(gpu):
                    proc.start()


class MultiThreadPrefetchData(DataFlow):
    """
    Create multiple dataflow instances and run them each in one thread.
    Collect outputs with a queue.
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
                for dp in self.df.get_data():
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
            get_df ( -> DataFlow): a callable which returns a DataFlow
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

    def size(self):
        return self.threads[0].size()

    def get_data(self):
        while True:
            yield self.queue.get()

    def __del__(self):
        for p in self.threads:
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

    def get_data(self):
        for dp in self.ds.get_data():
            oid = self.client.put(dp)
            yield [oid.binary()]


class PlasmaGetData(ProxyDataFlow):
    """
    Take plasma object id from a DataFlow, and retrieve it from plasma shared
    memory object store.
    """
    def __init__(self, ds, socket="/tmp/plasma"):
        self._socket = socket
        super(PlasmaGetData, self).__init__(ds)

    def reset_state(self):
        super(PlasmaGetData, self).reset_state()
        self.client = plasma.connect(self._socket, "", 0)

    def get_data(self):
        for dp in self.ds.get_data():
            oid = plasma.ObjectID(dp[0])
            dp = self.client.get(oid)
            yield dp


try:
    import pyarrow.plasma as plasma
except ImportError:
    from ..utils.develop import create_dummy_class
    PlasmaPutData = create_dummy_class('PlasmaPutData', 'pyarrow')   # noqa
    PlasmaGetData = create_dummy_class('PlasmaGetData', 'pyarrow')   # noqa
