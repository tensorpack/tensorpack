# -*- coding: UTF-8 -*-
# File: prefetch.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

from __future__ import print_function
import multiprocessing as mp
import itertools
from six.moves import range, zip, queue
import uuid
import os
import zmq

from .base import ProxyDataFlow
from .common import RepeatedData
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
        This is significantly slower than :class:`PrefetchDataZMQ` when data
        is large.
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

    Note that this dataflow is not fork-safe. You cannot nest this dataflow
    into another PrefetchDataZMQ or PrefetchData.
    """
    def __init__(self, ds, nr_proc=1, pipedir=None, hwm=50):
        """
        Args:
            ds (DataFlow): input DataFlow.
            nr_proc (int): number of processes to use.
            pipedir (str): a local directory where the pipes should be put.
                Useful if you're running on non-local FS such as NFS or GlusterFS.
            hwm (int): the zmq "high-water mark" for both sender and receiver.
        """
        super(PrefetchDataZMQ, self).__init__(ds)
        try:
            self._size = ds.size()
        except NotImplementedError:
            self._size = -1
        self.nr_proc = nr_proc

        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PULL)

        if pipedir is None:
            pipedir = os.environ.get('TENSORPACK_PIPEDIR', '.')
        assert os.path.isdir(pipedir), pipedir
        self.pipename = "ipc://{}/dataflow-pipe-".format(pipedir.rstrip('/')) + str(uuid.uuid1())[:6]
        self.socket.set_hwm(hwm)
        self.socket.bind(self.pipename)

        self.procs = [PrefetchProcessZMQ(self.ds, self.pipename, hwm)
                      for _ in range(self.nr_proc)]
        self.start_processes()
        # __del__ not guranteed to get called at exit
        import atexit
        atexit.register(lambda x: x.__del__(), self)

    def start_processes(self):
        start_proc_mask_signal(self.procs)

    def get_data(self):
        try:
            for k in itertools.count():
                if self._size > 0 and k >= self._size:
                    break
                dp = loads(self.socket.recv(copy=False).bytes)
                yield dp
        except zmq.ContextTerminated:
            logger.info("ContextTerminated in Master Prefetch Process")
            return
        except:
            raise

    def reset_state(self):
        # do nothing. all ds are reset once and only once in spawned processes
        pass

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
    Prefetch with each process having its own ``CUDA_VISIBLE_DEVICES`` variable
    mapped to one GPU.
    """

    def __init__(self, ds, gpus, pipedir=None):
        """
        Args:
            ds (DataFlow): input DataFlow.
            gpus (list[int]): list of GPUs to use. Will also start this many
                of processes.
            pipedir (str): a local directory where the pipes should be put.
                Useful if you're running on non-local FS such as NFS or GlusterFS.
        """
        self.gpus = gpus
        super(PrefetchOnGPUs, self).__init__(ds, len(gpus), pipedir)

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

    With threads, there are tiny communication overhead, but due to GIL, you
    should avoid starting the threads in your main process.
    Note that the threads will only start in the process which calls
    `reset_state()`.
    """
    class WorkerThread(StoppableThread):
        def __init__(self, inq, outq, map_func):
            self.inq = inq
            self.outq = outq
            self.func = map_func

        def run(self):
            while not self.stopped():
                dp = self.queue_get_stoppable(self.inq)
                dp = self.func(dp)
                if dp is not None:
                    self.queue_put_stoppable(self.outq, dp)

    def __init__(self, ds, nr_thread, map_func, buffer_size=200):
        """
        Args:
            pass
        """
        super(ThreadedMapData, self).__init__(ds)
        self.infinite_ds = RepeatedData(ds, -1)
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
        self._threads = [ThreadedMapData.WorkerThread(
            self._in_queue, self._out_queue, self.map_func)
            for _ in range(self.nr_thread)]
        for t in self._threads:
            t.start()

        # fill the buffer
        self._itr = self.infinite_ds.get_data()
        self._fill_buffer()

    def _fill_buffer(self):
        n = self.buffer_size - self._in_queue.qsize() - self._out_queue.qsize()
        if n <= 0:
            return
        for _ in range(n):
            self._in_queue.put(next(self._itr))

    def get_data(self):
        self._fill_buffer()
        sz = self.size()
        for _ in range(sz):
            self._in_queue.put(next(self._itr))
            yield self._out_queue.get()
