# -*- coding: UTF-8 -*-
# File: concurrency.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>
# Credit belongs to Xinyu Zhou

import threading
import multiprocessing
import atexit
import bisect
import weakref
import six
if six.PY2:
    import subprocess32 as subprocess
else:
    import subprocess

from . import logger

__all__ = ['StoppableThread', 'LoopThread', 'ensure_proc_terminate',
           'OrderedResultGatherProc', 'OrderedContainer', 'DIE']

class StoppableThread(threading.Thread):
    def __init__(self):
        super(StoppableThread, self).__init__()
        self._stop = threading.Event()

    def stop(self):
        self._stop.set()

    def stopped(self):
        return self._stop.isSet()

class LoopThread(threading.Thread):
    """ A pausable thread that simply runs a loop"""
    def __init__(self, func):
        """
        :param func: the function to run
        """
        super(LoopThread, self).__init__()
        self.func = func
        self.lock = threading.Lock()
        self.daemon = True

    def run(self):
        while True:
            self.lock.acquire()
            self.lock.release()
            self.func()

    def pause(self):
        self.lock.acquire()

    def resume(self):
        self.lock.release()


class DIE(object):
    """ A placeholder class indicating end of queue """
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

    assert isinstance(proc, multiprocessing.Process)
    atexit.register(stop_proc_by_weak_ref, weakref.ref(proc))

def subproc_call(cmd, timeout=None):
    try:
        output = subprocess.check_output(
                cmd, stderr=subprocess.STDOUT,
                shell=True, timeout=timeout)
        return output
    except subprocess.TimeoutExpired as e:
        logger.warn("Timeout in evaluation!")
        logger.warn(e.output)
    except subprocess.CalledProcessError as e:
        logger.warn("Evaluation script failed: {}".format(e.returncode))
        logger.warn(e.output)

class OrderedContainer(object):
    """
    Like a priority queue, but will always wait for item with index (x+1) before producing (x+2).
    """
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
    """
    Gather indexed data from a data queue, and produce results with the
    original index-based order.
    """
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
