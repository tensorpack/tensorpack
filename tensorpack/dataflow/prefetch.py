# -*- coding: UTF-8 -*-
# File: prefetch.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import multiprocessing

from six.moves import range
from .base import ProxyDataFlow
from ..utils.concurrency import ensure_procs_terminate

__all__ = ['PrefetchData']

class PrefetchProcess(multiprocessing.Process):
    def __init__(self, ds, queue):
        """
        :param ds: ds to take data from
        :param queue: output queue to put results in
        """
        super(PrefetchProcess, self).__init__()
        self.ds = ds
        self.queue = queue

    def run(self):
        # reset RNG of ds so each process will produce different data
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
        :param nr_proc: number of processes to use.
        """
        super(PrefetchData, self).__init__(ds)
        self._size = self.size()
        self.nr_proc = nr_proc
        self.nr_prefetch = nr_prefetch
        self.queue = multiprocessing.Queue(self.nr_prefetch)
        self.procs = [PrefetchProcess(self.ds, self.queue)
                      for _ in range(self.nr_proc)]
        ensure_procs_terminate(self.procs)
        for x in self.procs:
            x.start()

    def get_data(self):
        tot_cnt = 0
        for _ in range(tot_cnt):
            dp = self.queue.get()
            yield dp

    def __del__(self):
        self.queue.close()
        for x in self.procs:
            x.terminate()

