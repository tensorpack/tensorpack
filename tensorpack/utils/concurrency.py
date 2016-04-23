# -*- coding: UTF-8 -*-
# File: concurrency.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>
# Credit belongs to Xinyu Zhou

import threading
import multiprocessing, multiprocess
from contextlib import contextmanager
import tensorflow as tf
import atexit
import weakref
from six.moves import zip

from .naming import *

__all__ = ['StoppableThread', 'ensure_proc_terminate',
           'OrderedResultGatherProc', 'OrderedContainer', 'DIE']

class StoppableThread(threading.Thread):
    def __init__(self):
        super(StoppableThread, self).__init__()
        self._stop = threading.Event()

    def stop(self):
        self._stop.set()

    def stopped(self):
        return self._stop.isSet()


class DIE(object):
    pass


def ensure_proc_terminate(proc):
    if isinstance(proc, list):
        for p in proc:
            ensure_proc_terminate(p)
        return

    def stop_proc_by_weak_ref(ref):
        proc = ref()
        if proc is None:
            return
        if not proc.is_alive():
            return
        proc.terminate()
        proc.join()

    assert isinstance(proc, (multiprocessing.Process, multiprocess.Process))
    atexit.register(stop_proc_by_weak_ref, weakref.ref(proc))


class OrderedContainer(object):
    def __init__(self, start=0):
        self.ranks = []
        self.data = []
        self.wait_for = start

    def put(self, rank, val):
        idx = bisect.bisect(self.ranks, rank)
        self.ranks.insert(idx, rank)
        self.data.insert(idx, val)

    def has_next(self):
        if len(self.ranks) == 0:
            return False
        return self.ranks[0] == self.wait_for

    def get(self):
        assert self.has_next()
        ret = self.data[0]
        rank = self.ranks[0]
        del self.ranks[0]
        del self.data[0]
        self.wait_for += 1
        return rank, ret


class OrderedResultGatherProc(multiprocessing.Process):
    def __init__(self, data_queue, start=0):
        super(self.__class__, self).__init__()

        self.data_queue = data_queue
        self.ordered_container = OrderedContainer(start=start)
        self.result_queue = multiprocessing.Queue()

    def run(self):
        try:
            while True:
                task_id, data = self.data_queue.get()
                if task_id == DIE:
                    self.result_queue.put((task_id, data))
                else:
                    self.ordered_container.put(task_id, data)
                    while self.ordered_container.has_next():
                        self.result_queue.put(self.ordered_container.get())
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise e

    def get(self):
        return self.result_queue.get()
