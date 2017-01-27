#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: graph.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

""" Graph related callbacks"""

from .base import Triggerable

__all__ = ['RunOp']


class RunOp(Triggerable):
    """ Run an Op. """

    def __init__(self, setup_func, run_before=True, run_epoch=True):
        """
        Args:
            setup_func: a function that returns the Op in the graph
            run_before (bool): run the Op before training
            run_epoch (bool): run the Op on every epoch trigger

        Examples:
            The `DQN Example
            <https://github.com/ppwwyyxx/tensorpack/blob/master/examples/Atari2600/DQN.py#L182>`_
            uses this callback to update target network.
        """
        self.setup_func = setup_func
        self.run_before = run_before
        self.run_epoch = run_epoch

    def _setup_graph(self):
        self._op = self.setup_func()

    def _before_train(self):
        if self.run_before:
            self._op.run()

    def _trigger(self):
        if self.run_epoch:
            self._op.run()
