# -*- coding: UTF-8 -*-
# File: prefetch.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import multiprocessing as mp
from threading import Thread
import itertools
from six.moves import range, zip
from six.moves.queue import Queue
import uuid
import os

from .base import ProxyDataFlow
from ..utils.concurrency import *
from ..utils.serialize import loads, dumps
from ..utils import logger, change_env

try:
    import zmq
except ImportError:
    logger.warn("Error in 'import zmq'. PrefetchDataZMQ won't be available.")
    __all__ = ['PrefetchData', 'BlockParallel']
else:
    __all__.extend(['PrefetchDataZMQ', 'PrefetchOnGPUs'])


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
    Prefetch data from a `DataFlow` using multiprocessing
    """
    def __init__(self, ds, nr_prefetch, nr_proc=1):
        """
        :param ds: a `DataFlow` instance.
        :param nr_prefetch: size of the queue to hold prefetched datapoints.
        :param nr_proc: number of processes to use. When larger than 1, order
            of data points will be random.
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
        for x in self.procs:
            x.start()

    def get_data(self):
        for k in itertools.count():
            if self._size > 0 and k >= self._size:
                break
            dp = self.queue.get()
            yield dp

    def reset_state(self):
        # do nothing. all ds are reset once and only once in spawned processes
        pass

def BlockParallel(ds, queue_size):
    """
    Insert `BlockParallel` in dataflow pipeline to block parallelism on ds

    :param ds: a `DataFlow`
    :param queue_size: size of the queue used
    """
    return PrefetchData(ds, queue_size, 1)

class PrefetchProcessZMQ(mp.Process):
    def __init__(self, ds, conn_name):
        """
        :param ds: a `DataFlow` instance.
        :param conn_name: the name of the IPC connection
        """
        super(PrefetchProcessZMQ, self).__init__()
        self.ds = ds
        self.conn_name = conn_name

    def run(self):
        self.ds.reset_state()
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUSH)
        self.socket.set_hwm(1)
        self.socket.connect(self.conn_name)
        while True:
            for dp in self.ds.get_data():
                self.socket.send(dumps(dp), copy=False)

class PrefetchDataZMQ(ProxyDataFlow):
    """ Work the same as `PrefetchData`, but faster. """
    def __init__(self, ds, nr_proc=1, pipedir=None):
        """
        :param ds: a `DataFlow` instance.
        :param nr_proc: number of processes to use. When larger than 1, order
            of datapoints will be random.
        :param pipedir: a local directory where the pipes would be. Useful if you're running on non-local FS such as NFS.
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
        self.socket.set_hwm(5)  # a little bit faster than default, don't know why
        self.socket.bind(self.pipename)

        self.procs = [PrefetchProcessZMQ(self.ds, self.pipename)
                      for _ in range(self.nr_proc)]
        self.start_processes()
        # __del__ not guranteed to get called at exit
        import atexit
        atexit.register(lambda x: x.__del__(), self)

    def start_processes(self):
        start_proc_mask_signal(self.procs)

    def get_data(self):
        for k in itertools.count():
            if self._size > 0 and k >= self._size:
                break
            dp = loads(self.socket.recv(copy=False))
            yield dp

    def reset_state(self):
        # do nothing. all ds are reset once and only once in spawned processes
        pass

    def __del__(self):
        # on exit, logger may not be functional anymore
        try:
            logger.info("Prefetch process exiting...")
        except:
            pass
        if not self.context.closed:
            self.context.destroy(0)
        for x in self.procs:
            x.terminate()
        try:
            logger.info("Prefetch process exited.")
        except:
            pass

class PrefetchOnGPUs(PrefetchDataZMQ):
    """ Prefetch with each process having a specific CUDA_VISIBLE_DEVICES"""
    def __init__(self, ds, gpus, pipedir=None):
        super(PrefetchOnGPUs, self).__init__(ds, len(gpus), pipedir)
        self.gpus = gpus

    def start_processes(self):
        with mask_sigint():
            for gpu, proc in zip(self.gpus, self.procs):
                with change_gpu(gpu):
                    proc.start()

