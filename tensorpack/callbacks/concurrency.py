#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: concurrency.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

from .base import Callback
from ..utils.concurrency import start_proc_mask_signal
from ..utils import logger

__all__ = ['StartProcOrThread']

class StartProcOrThread(Callback):
    def __init__(self, procs_threads):
        """
        Start extra threads and processes before training
        :param procs_threads: list of processes or threads
        """
        if not isinstance(procs_threads, list):
            procs_threads = [procs_threads]
        self._procs_threads = procs_threads

    def _before_train(self):
        logger.info("Starting " +  \
                ', '.join([k.name for k in self._procs_threads]))
        # avoid sigint get handled by other processes
        start_proc_mask_signal(self._procs_threads)
