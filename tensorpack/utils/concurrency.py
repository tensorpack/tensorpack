# -*- coding: UTF-8 -*-
# File: concurrency.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>
# Credit belongs to Xinyu Zhou

import threading
import multiprocessing
import atexit
import bisect
from contextlib import contextmanager
import signal
import weakref
import six
from six.moves import queue

from . import logger

if six.PY2:
    import subprocess32 as subprocess
else:
    import subprocess


__all__ = ['StoppableThread', 'LoopThread', 'ShareSessionThread',
           'ensure_proc_terminate',
           'OrderedResultGatherProc', 'OrderedContainer', 'DIE',
           'mask_sigint', 'start_proc_mask_signal']


class StoppableThread(threading.Thread):
    """
    A thread that has a 'stop' event.
    """

    def __init__(self, evt=None):
        """
        Args:
            evt(threading.Event): if None, will create one.
        """
        super(StoppableThread, self).__init__()
        if evt is None:
            evt = threading.Event()
        self._stop_evt = evt

    def stop(self):
        """ Stop the thread"""
        self._stop_evt.set()

    def stopped(self):
        """
        Returns:
            bool: whether the thread is stopped or not
        """
        return self._stop_evt.isSet()

    def queue_put_stoppable(self, q, obj):
        """ Put obj to queue, but will give up when the thread is stopped"""
        while not self.stopped():
            try:
                q.put(obj, timeout=5)
                break
            except queue.Full:
                pass

    def queue_get_stoppable(self, q):
        """ Take obj from queue, but will give up when the thread is stopped"""
        while not self.stopped():
            try:
                return q.get(timeout=5)
            except queue.Empty:
                pass


class LoopThread(StoppableThread):
    """ A pausable thread that simply runs a loop"""

    def __init__(self, func, pausable=True):
        """
        Args:
            func: the function to run
        """
        super(LoopThread, self).__init__()
        self._func = func
        self._pausable = pausable
        if pausable:
            self._lock = threading.Lock()
        self.daemon = True

    def run(self):
        while not self.stopped():
            if self._pausable:
                self._lock.acquire()
                self._lock.release()
            self._func()

    def pause(self):
        """ Pause the loop """
        assert self._pausable
        self._lock.acquire()

    def resume(self):
        """ Resume the loop """
        assert self._pausable
        self._lock.release()


class ShareSessionThread(threading.Thread):
    """ A wrapper around thread so that the thread
        uses the default session at "start()" time.
    """
    def __init__(self, th=None):
        """
        Args:
            th (threading.Thread or None):
        """
        super(ShareSessionThread, self).__init__()
        if th is not None:
            assert isinstance(th, threading.Thread), th
            self._th = th
            self.name = th.name
            self.daemon = th.daemon

    @contextmanager
    def default_sess(self):
        if self._sess:
            with self._sess.as_default():
                yield
        else:
            logger.warn("ShareSessionThread {} wasn't under a default session!".format(self.name))
            yield

    def start(self):
        import tensorflow as tf
        self._sess = tf.get_default_session()
        super(ShareSessionThread, self).start()

    def run(self):
        if not self._th:
            raise NotImplementedError()
        with self._sess.as_default():
            self._th.run()


class DIE(object):
    """ A placeholder class indicating end of queue """
    pass


def ensure_proc_terminate(proc):
    """
    Make sure processes terminate when main process exit.

    Args:
        proc (multiprocessing.Process or list)
    """
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


def is_main_thread():
    if six.PY2:
        return isinstance(threading.current_thread(), threading._MainThread)
    else:
        # a nicer solution with py3
        return threading.current_thread() == threading.main_thread()


@contextmanager
def mask_sigint():
    """
    Returns:
        If called in main thread, returns a context where ``SIGINT`` is ignored, and yield True.
        Otherwise yield False.
    """
    if is_main_thread():
        sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
        yield True
        signal.signal(signal.SIGINT, sigint_handler)
    else:
        yield False


def start_proc_mask_signal(proc):
    """
    Start process(es) with SIGINT ignored.

    Args:
        proc: (multiprocessing.Process or list)

    Note:
        The signal mask is only applied when called from main thread.
    """
    if not isinstance(proc, list):
        proc = [proc]

    with mask_sigint():
        for p in proc:
            p.start()


def subproc_call(cmd, timeout=None):
    """
    Execute a command with timeout, and return both STDOUT/STDERR.

    Args:
        cmd(str): the command to execute.
        timeout(float): timeout in seconds.

    Returns:
        output(bytes), retcode(int). If timeout, retcode is -1.
    """
    try:
        output = subprocess.check_output(
            cmd, stderr=subprocess.STDOUT,
            shell=True, timeout=timeout)
        return output, 0
    except subprocess.TimeoutExpired as e:
        logger.warn("Command timeout!")
        logger.warn(e.output.decode('utf-8'))
        return e.output, -1
    except subprocess.CalledProcessError as e:
        logger.warn("Command failed: {}".format(e.returncode))
        logger.warn(e.output.decode('utf-8'))
        return e.output, e.returncode
    except Exception:
        logger.warn("Command failed to run: {}".format(cmd))
        return "", -2


class OrderedContainer(object):
    """
    Like a queue, but will always wait to receive item with rank
    (x+1) and produce (x+1) before producing (x+2).

    Warning:
        It is not thread-safe.
    """

    def __init__(self, start=0):
        """
        Args:
            start(int): the starting rank.
        """
        self.ranks = []
        self.data = []
        self.wait_for = start

    def put(self, rank, val):
        """
        Args:
            rank(int): rank of th element. All elements must have different ranks.
            val: an object
        """
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

    def __init__(self, data_queue, nr_producer, start=0):
        """
        Args:
            data_queue(multiprocessing.Queue): a queue which contains datapoints.
            nr_producer(int): number of producer processes. This process will
                terminate after receiving this many of :class:`DIE` sentinel.
            start(int): the rank of the first object
        """
        super(OrderedResultGatherProc, self).__init__()
        self.data_queue = data_queue
        self.ordered_container = OrderedContainer(start=start)
        self.result_queue = multiprocessing.Queue()
        self.nr_producer = nr_producer

    def run(self):
        nr_end = 0
        try:
            while True:
                task_id, data = self.data_queue.get()
                if task_id == DIE:
                    self.result_queue.put((task_id, data))
                    nr_end += 1
                    if nr_end == self.nr_producer:
                        return
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
