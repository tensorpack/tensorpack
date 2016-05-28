#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: graph.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

""" Graph related callbacks"""

from .base import Callback
from ..utils import logger

__all__ = ['RunOp']

class RunOp(Callback):
    """ Run an op periodically"""
    def __init__(self, setup_func, run_before=True, run_epoch=True):
        """
        :param setup_func: a function that returns the op in the graph
        :param run_before: run the op before training
        :param run_epoch: run the op on every epoch trigger
        """
        self.setup_func = setup_func
        self.run_before = run_before
        self.run_epoch = run_epoch

    def _setup_graph(self):
        self._op = self.setup_func()
        #self._op_name = self._op.name

    def _before_train(self):
        if self.run_before:
            self._op.run()

    def _trigger_epoch(self):
        if self.run_epoch:
            self._op.run()

    #def _log(self):
        #logger.info("Running op {} ...".format(self._op_name))
