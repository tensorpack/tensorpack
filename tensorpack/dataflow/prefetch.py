# -*- coding: UTF-8 -*-
# File: prefetch.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

from __future__ import print_function
import multiprocessing as mp
import itertools
from six.moves import range, zip, queue
import errno
import uuid
import os
import zmq

from .base import ProxyDataFlow, DataFlowTerminated
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

    Note:
        1. This is significantly slower than :class:`PrefetchDataZMQ` when data is large.
        2. When nesting like this: ``PrefetchDataZMQ(PrefetchData(df, nr_proc=a), nr_proc=b)``.
           A total of ``a`` instances of ``df`` worker processes will be created.
           This is different from the behavior of :class`PrefetchDataZMQ`
        3. The underlying dataflow worker will be forked multiple times When ``nr_proc>1``.
           As a result, unless the underlying dataflow is fully shuffled, the data distribution
           produced by this dataflow will be wrong.
           (e.g. you are likely to see duplicated datapoints at the beginning)
    """
    def __init__(self, ds, nr_prefetch, nr_proc=1):
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
        self.queue = mp.Queue(self.nr_prefetch)
        self.procs = [PrefetchProcess(self.ds, self.queue)
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
        while True:
            for dp in self.ds.get_data():
                self.socket.send(dumps(dp), copy=False)


class PrefetchDataZMQ(ProxyDataFlow):
    """
    Prefetch data from a DataFlow using multiple processes, with ZMQ for
    communication.

    A local directory is needed to put the ZMQ pipes.
    You can set this with env var ``$TENSORPACK_PIPEDIR`` if you're running on non-local FS such as NFS or GlusterFS.

    Note:
        1. Once :meth:`reset_state` is called, this dataflow becomes not fork-safe.
        2. When nesting like this: ``PrefetchDataZMQ(PrefetchDataZMQ(df, a), b)``.
           A total of ``a * b`` instances of ``df`` worker processes will be created.
        3. The underlying dataflow worker will be forked multiple times When ``nr_proc>1``.
           As a result, unless the underlying dataflow is fully shuffled, the data distribution
           produced by this dataflow will be wrong.
           (e.g. you are likely to see duplicated datapoints at the beginning)
    """
    def __init__(self, ds, nr_proc=1, hwm=50):
        """
        Args:
            ds (DataFlow): input DataFlow.
            nr_proc (int): number of processes to use.
            hwm (int): the zmq "high-water mark" for both sender and receiver.
        """
        assert os.name != 'nt', "PrefetchDataZMQ doesn't support windows!  Consider PrefetchData instead."
        super(PrefetchDataZMQ, self).__init__(ds)
        try:
            self._size = ds.size()
        except NotImplementedError:
            self._size = -1
        self.nr_proc = nr_proc
        self._hwm = hwm
        self._finish_setup = False

    def get_data(self):
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
        if self._finish_setup:
            return
        self._finish_setup = True

        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PULL)

        pipedir = os.environ.get('TENSORPACK_PIPEDIR', '.')
        assert os.path.isdir(pipedir), pipedir
        self.pipename = "ipc://{}/dataflow-pipe-".format(pipedir.rstrip('/')) + str(uuid.uuid1())[:6]
        self.socket.set_hwm(self._hwm)
        self.socket.bind(self.pipename)

        self.procs = [PrefetchProcessZMQ(self.ds, self.pipename, self._hwm)
                      for _ in range(self.nr_proc)]
        self.start_processes()
        # __del__ not guranteed to get called at exit
        import atexit
        atexit.register(lambda x: x.__del__(), self)

    def start_processes(self):
        start_proc_mask_signal(self.procs)

    def __del__(self):
        # on exit, logger may not be functional anymore
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

    def start_processes(self):
        with mask_sigint():
            for gpu, proc in zip(self.gpus, self.procs):
                with change_gpu(gpu):
                    proc.start()


class ThreadedMapData(ProxyDataFlow):
    """
    Same as :class:`MapData`, but start threads to run the mapping function.
    This is useful when the mapping function is the bottleneck, but you don't
    want to start processes for the entire dataflow pipeline.

    Notes:
        1. There is tiny communication overhead with threads, but you
        should avoid starting many threads in your main process to avoid GIL.

        The threads will only start in the process which calls :meth:`reset_state()`.
        Therefore you can use ``PrefetchDataZMQ(ThreadedMapData(...), 1)`` to avoid GIL.

        2. Threads run in parallel and can take different time to run the
           mapping function. Therefore the order of datapoints won't be
           preserved, and datapoints from one pass of `df.get_data()` might get
           mixed with datapoints from the next pass.

           You can use **strict mode**, where `ThreadedMapData.get_data()`
           is guranteed to produce the exact set which `df.get_data()`
           produces. Although the order of data still isn't preserved.
    """
    class _WorkerThread(StoppableThread):
        def __init__(self, inq, outq, map_func, strict):
            super(ThreadedMapData._WorkerThread, self).__init__()
            self.inq = inq
            self.outq = outq
            self.func = map_func
            self.daemon = True
            self._strict = strict

        def run(self):
            while not self.stopped():
                dp = self.queue_get_stoppable(self.inq)
                dp = self.func(dp)
                if dp is not None:
                    self.queue_put_stoppable(self.outq, dp)
                else:
                    assert not self._strict, \
                        "[ThreadedMapData] Map function cannot return None when strict mode is used."

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

    def reset_state(self):
        super(ThreadedMapData, self).reset_state()
        for t in self._threads:
            t.stop()
            t.join()
        self._in_queue = queue.Queue()
        self._out_queue = queue.Queue()
        self._threads = [ThreadedMapData._WorkerThread(
            self._in_queue, self._out_queue, self.map_func, self._strict)
            for _ in range(self.nr_thread)]
        for t in self._threads:
            t.start()

        self._iter = self._iter_ds.get_data()

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
