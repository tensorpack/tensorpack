#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: prefetch.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>


from .base import DataFlow
import multiprocessing

__all__ = ['PrefetchData']

class PrefetchProcess(multiprocessing.Process):
    def __init__(self, ds, queue_size):
        super(PrefetchProcess, self).__init__()
        self.ds = ds
        self.queue = multiprocessing.Queue(queue_size)
        class Sentinel:
            pass
        self.sentinel = Sentinel()

    def run(self):
        for dp in self.ds.get_data():
            self.queue.put(dp)
        self.queue.put(self.sentinel)

    def get_data(self):
        while True:
            ret = self.queue.get()
            if ret is self.sentinel:
                return
            yield ret

class PrefetchData(DataFlow):
    def __init__(self, ds, nr_prefetch):
        self.ds = ds
        self.nr_prefetch = int(nr_prefetch)
        assert self.nr_prefetch > 0

    def size(self):
        return self.ds.size()

    def get_data(self):
        worker = PrefetchProcess(self.ds, self.nr_prefetch)
        # TODO register terminate function
        worker.start()
        for dp in worker.get_data():
            yield dp
        worker.join()
        worker.terminate()

