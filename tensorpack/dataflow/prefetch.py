# -*- coding: UTF-8 -*-
# File: prefetch.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

from __future__ import print_function
import threading
import multiprocessing as mp
import itertools
from six.moves import range, zip, queue
import errno
import uuid
import os
import zmq

from .base import ProxyDataFlow, DataFlowTerminated, DataFlowReentrantGuard
from ..utils.concurrency import (ensure_proc_terminate,
                                 mask_sigint, start_proc_mask_signal,
                                 StoppableThread)
from ..utils.serialize import loads, dumps
from ..utils import logger
from ..utils.gpu import change_gpu

__all__ = ['PrefetchData', 'PrefetchDataZMQ', 'PrefetchOnGPUs',
           'ThreadedMapData']


class PrefetchProcess(mp.Process):
    def __init__(self, ds, queue, reset_after_spawn=True):
        """
        :param ds: ds to take data from
        :param queue: output queue to put results in
        """
        super(PrefetchProcess, self).__init__()
        self.ds = ds
        self.queue = queue
        self.reset_after_spawn = reset_after_spawn

    def run(self):
        if self.reset_after_spawn:
            # reset all ds so each process will produce different data
            self.ds.reset_state()
        while True:
            for dp in self.ds.get_data():
                self.queue.put(dp)


class PrefetchData(ProxyDataFlow):
    """
    Prefetch data from a DataFlow using Python multiprocessing utilities.
    It will fork the process calling :meth:`__init__`, collect datapoints from `ds` in each
    process by a Python :class:`multiprocessing.Queue`.

    Note:
        1. The underlying dataflow worker will be forked multiple times when ``nr_proc>1``.
           As a result, unless the underlying dataflow is fully shuffled, the data distribution
           produced by this dataflow will be different.
           (e.g. you are likely to see duplicated datapoints at the beginning)
        2. This is significantly slower than :class:`PrefetchDataZMQ` when data is large.
        3. When nesting like this: ``PrefetchDataZMQ(PrefetchData(df, nr_proc=a), nr_proc=b)``.
           A total of ``a`` instances of ``df`` worker processes will be created.
           This is different from the behavior of :class:`PrefetchDataZMQ`
        4. `reset_state()` is a no-op. The worker processes won't get called.
    """
    def __init__(self, ds, nr_prefetch, nr_proc):
        """
        Args:
            ds (DataFlow): input DataFlow.
            nr_prefetch (int): size of the queue to hold prefetched datapoints.
            nr_proc (int): number of processes to use.
        """
        super(PrefetchData, self).__init__(ds)
        try:
            self._size = ds.size()
        except NotImplementedError:
            self._size = -1
        self.nr_proc = nr_proc
        self.nr_prefetch = nr_prefetch
        self._guard = DataFlowReentrantGuard()

        self.queue = mp.Queue(self.nr_prefetch)
        self.procs = [PrefetchProcess(self.ds, self.queue)
                      for _ in range(self.nr_proc)]
        ensure_proc_terminate(self.procs)
        start_proc_mask_signal(self.procs)

    def get_data(self):
        with self._guard:
            for k in itertools.count():
                if self._size > 0 and k >= self._size:
                    break
                dp = self.queue.get()
                yield dp

    def reset_state(self):
        # do nothing. all ds are reset once and only once in spawned processes
        pass


class PrefetchProcessZMQ(mp.Process):
    def __init__(self, ds, conn_name, hwm):
        super(PrefetchProcessZMQ, self).__init__()
        self.ds = ds
        self.conn_name = conn_name
        self.hwm = hwm

    def run(self):
        self.ds.reset_state()
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUSH)
        self.socket.set_hwm(self.hwm)
        self.socket.connect(self.conn_name)
        try:
            while True:
                for dp in self.ds.get_data():
                    self.socket.send(dumps(dp), copy=False)
        # sigint could still propagate here, e.g. when nested
        except KeyboardInterrupt:
            pass


class PrefetchDataZMQ(ProxyDataFlow):
    """
    Prefetch data from a DataFlow using multiple processes, with ZeroMQ for
    communication.
    It will fork the process calling :meth:`reset_state()`,
    collect datapoints from `ds` in each process by ZeroMQ IPC pipe.

    Note:
        1. The underlying dataflow worker will be forked multiple times When ``nr_proc>1``.
           As a result, unless the underlying dataflow is fully shuffled, the data distribution
           produced by this dataflow will be different.
           (e.g. you are likely to see duplicated datapoints at the beginning)
        2. Once :meth:`reset_state` is called, this dataflow becomes not fork-safe.
           i.e., if you fork an already reset instance of this dataflow,
           it won't be usable in the forked process.
        3. When nesting like this: ``PrefetchDataZMQ(PrefetchDataZMQ(df, nr_proc=a), nr_proc=b)``.
           A total of ``a * b`` instances of ``df`` worker processes will be created.
           Also in this case, some zmq pipes cannot be cleaned at exit.
        4. By default, a UNIX named pipe will be created in the current directory.
           However, certain non-local filesystem such as NFS/GlusterFS/AFS doesn't always support pipes.
           You can change the directory by ``export TENSORPACK_PIPEDIR=/other/dir``.
           In particular, you can use somewhere under '/tmp' which is usually local.

           Note that some non-local FS may appear to support pipes and code
           may appear to run but crash with bizarre error.
           Also note that ZMQ limits the maximum length of pipe path.
           If you hit the limit, you can set the directory to a softlink
           which points to a local directory.
        5. Calling `reset_state()` more than once is a no-op, i.e. the worker processes won't get called.
    """
    def __init__(self, ds, nr_proc=1, hwm=50):
        """
        Args:
            ds (DataFlow): input DataFlow.
            nr_proc (int): number of processes to use.
            hwm (int): the zmq "high-water mark" (queue size) for both sender and receiver.
        """
        assert os.name != 'nt', "PrefetchDataZMQ doesn't support windows!  PrefetchData might work sometimes."
        super(PrefetchDataZMQ, self).__init__(ds)
        try:
            self._size = ds.size()
        except NotImplementedError:
            self._size = -1
        self.nr_proc = nr_proc
        self._hwm = hwm

        self._guard = DataFlowReentrantGuard()
        self._setup_done = False

    def get_data(self):
        with self._guard:
            try:
                for k in itertools.count():
                    if self._size > 0 and k >= self._size:
                        break
                    dp = loads(self.socket.recv(copy=False).bytes)
                    yield dp
            except zmq.ContextTerminated:
                logger.info("[Prefetch Master] Context terminated.")
                raise DataFlowTerminated()
            except zmq.ZMQError as e:
                if e.errno == errno.ENOTSOCK:       # socket closed
                    logger.info("[Prefetch Master] Socket closed.")
                    raise DataFlowTerminated()
                else:
                    raise
            except:
                raise

    def reset_state(self):
        """
        All forked dataflows are reset **once and only once** in spawned processes.
        Nothing more can be done when calling this method.
        """
        if self._setup_done:
            return
        self._setup_done = True

        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PULL)

        pipedir = os.environ.get('TENSORPACK_PIPEDIR', '.')
        assert os.path.isdir(pipedir), pipedir
        self.pipename = "ipc://{}/dataflow-pipe-".format(pipedir.rstrip('/')) + str(uuid.uuid1())[:6]
        self.socket.set_hwm(self._hwm)
        try:
            self.socket.bind(self.pipename)
        except zmq.ZMQError:
            logger.error(
                "ZMQError in socket.bind(). Perhaps you're \
                using pipes on a non-local file system. See documentation of PrefetchDataZMQ for more information.")
            raise

        self.procs = [PrefetchProcessZMQ(self.ds, self.pipename, self._hwm)
                      for _ in range(self.nr_proc)]
        self._start_processes()
        # __del__ not guranteed to get called at exit
        import atexit
        atexit.register(lambda x: x.__del__(), self)

    def _start_processes(self):
        start_proc_mask_signal(self.procs)

    def __del__(self):
        if not self._setup_done:
            return
        if not self.context.closed:
            self.context.destroy(0)
        for x in self.procs:
            x.terminate()
        try:
            # TODO test if logger here would overwrite log file
            print("Prefetch process exited.")
        except:
            pass


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
            for gpu, proc in zip(self.gpus, self.procs):
                with change_gpu(gpu):
                    proc.start()


class ThreadedMapData(ProxyDataFlow):
    """
    Same as :class:`MapData`, but start threads to run the mapping function.
    This is useful when the mapping function is the bottleneck, but you don't
    want to start processes for the entire dataflow pipeline.

    Note:
        1. There is tiny communication overhead with threads, but you
           should avoid starting many threads in your main process to reduce GIL contention.

           The threads will only start in the process which calls :meth:`reset_state()`.
           Therefore you can use ``PrefetchDataZMQ(ThreadedMapData(...), 1)``
           to reduce GIL contention.

        2. Threads run in parallel and can take different time to run the
           mapping function. Therefore the order of datapoints won't be
           preserved, and datapoints from one pass of `df.get_data()` might get
           mixed with datapoints from the next pass.

           You can use **strict mode**, where `ThreadedMapData.get_data()`
           is guranteed to produce the exact set which `df.get_data()`
           produces. Although the order of data still isn't preserved.
    """
    class _WorkerThread(StoppableThread):
        def __init__(self, inq, outq, evt, map_func, strict):
            super(ThreadedMapData._WorkerThread, self).__init__(evt)
            self.inq = inq
            self.outq = outq
            self.func = map_func
            self.daemon = True
            self._strict = strict

        def run(self):
            try:
                while True:
                    dp = self.queue_get_stoppable(self.inq)
                    if self.stopped():
                        return

                    dp = self.func(dp)
                    if dp is not None:
                        self.outq.put(dp)
                    else:
                        assert not self._strict, \
                            "[ThreadedMapData] Map function cannot return None when strict mode is used."
            except:
                if self.stopped():
                    pass        # skip duplicated error messages
                else:
                    raise
            finally:
                self.stop()

    def __init__(self, ds, nr_thread, map_func, buffer_size=200, strict=False):
        """
        Args:
            ds (DataFlow): the dataflow to map
            nr_thread (int): number of threads to use
            map_func (callable): datapoint -> datapoint | None
            buffer_size (int): number of datapoints in the buffer
            strict (bool): use "strict mode", see notes above.
        """
        super(ThreadedMapData, self).__init__(ds)

        self._iter_ds = ds
        self._strict = strict
        self.nr_thread = nr_thread
        self.buffer_size = buffer_size
        self.map_func = map_func
        self._threads = []
        self._evt = None

    def reset_state(self):
        super(ThreadedMapData, self).reset_state()
        if self._threads:
            self._threads[0].stop()
            for t in self._threads:
                t.join()

        self._in_queue = queue.Queue()
        self._out_queue = queue.Queue()
        self._evt = threading.Event()
        self._threads = [ThreadedMapData._WorkerThread(
            self._in_queue, self._out_queue, self._evt, self.map_func, self._strict)
            for _ in range(self.nr_thread)]
        for t in self._threads:
            t.start()

        self._iter = self._iter_ds.get_data()
        self._guard = DataFlowReentrantGuard()

        # only call once, to ensure inq+outq has a total of buffer_size elements
        self._fill_buffer()

    def _fill_buffer(self):
        n = self.buffer_size - self._in_queue.qsize() - self._out_queue.qsize()
        assert n >= 0, n
        if n == 0:
            return
        try:
            for _ in range(n):
                self._in_queue.put(next(self._iter))
        except StopIteration:
            logger.error("[ThreadedMapData] buffer_size cannot be larger than the size of the DataFlow!")
            raise

    def get_data(self):
        with self._guard:
            for dp in self._iter:
                self._in_queue.put(dp)
                yield self._out_queue.get()

            self._iter = self._iter_ds.get_data()
            if self._strict:
                # first call get() to clear the queues, then fill
                for k in range(self.buffer_size):
                    dp = self._out_queue.get()
                    if k == self.buffer_size - 1:
                        self._fill_buffer()
                    yield dp
            else:
                for _ in range(self.buffer_size):
                    self._in_queue.put(next(self._iter))
                    yield self._out_queue.get()

    def __del__(self):
        if self._evt is not None:
            self._evt.set()
        for p in self._threads:
            p.join()
