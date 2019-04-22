# -*- coding: utf-8 -*-
# File: concurrency.py

import multiprocessing as mp

from ..utils import logger
from ..utils.concurrency import StoppableThread, start_proc_mask_signal
from .base import Callback

__all__ = ['StartProcOrThread']


class StartProcOrThread(Callback):
    """
    Start some threads or processes before training.
    """

    _chief_only = False

    def __init__(self, startable, stop_at_last=True):
        """
        Args:
            startable (list): list of processes or threads which have ``start()`` method.
                Can also be a single instance of process of thread.
            stop_at_last (bool): whether to stop the processes or threads
                after training. It will use :meth:`Process.terminate()` or
                :meth:`StoppableThread.stop()`, but will do nothing on normal
                ``threading.Thread`` or other startable objects.
        """
        if not isinstance(startable, list):
            startable = [startable]
        self._procs_threads = startable
        self._stop_at_last = stop_at_last

    def _before_train(self):
        logger.info("Starting " +
                    ', '.join([k.name for k in self._procs_threads]) + ' ...')
        # avoid sigint get handled by other processes
        start_proc_mask_signal(self._procs_threads)

    def _after_train(self):
        if not self._stop_at_last:
            return
        for k in self._procs_threads:
            if not k.is_alive():
                continue
            if isinstance(k, mp.Process):
                logger.info("Stopping {} ...".format(k.name))
                k.terminate()
                k.join(5.0)
                if k.is_alive():
                    logger.error("Cannot join process {}.".format(k.name))
            elif isinstance(k, StoppableThread):
                logger.info("Stopping {} ...".format(k.name))
                k.stop()
                k.join(5.0)
                if k.is_alive():
                    logger.error("Cannot join thread {}.".format(k.name))
