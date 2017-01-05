#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: concurrency.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

from .base import Callback
from ..utils.concurrency import start_proc_mask_signal
from ..utils import logger

__all__ = ['StartProcOrThread']


class StartProcOrThread(Callback):
    """
    Start some threads or processes before training.
    """

    def __init__(self, startable):
        """
        Args:
            startable(list): list of processes or threads which have ``start()`` method.
                Can also be a single instance of process of thread.
        """
        if not isinstance(startable, list):
            startable = [startable]
        self._procs_threads = startable

    def _before_train(self):
        logger.info("Starting " +
                    ', '.join([k.name for k in self._procs_threads]))
        # avoid sigint get handled by other processes
        start_proc_mask_signal(self._procs_threads)
