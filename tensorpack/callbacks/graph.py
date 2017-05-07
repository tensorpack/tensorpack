#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: graph.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

""" Graph related callbacks"""

import tensorflow as tf

from ..utils import logger
from .base import Callback

__all__ = ['RunOp', 'RunUpdateOps']


class RunOp(Callback):
    """ Run an Op. """

    def __init__(self, setup_func,
                 run_before=True, run_as_trigger=True, run_step=False):
        """
        Args:
            setup_func: a function that returns the Op in the graph
            run_before (bool): run the Op before training
            run_as_trigger (bool): run the Op on every trigger
            run_step (bool): run the Op every step (along with training)

        Examples:
            The `DQN Example
            <https://github.com/ppwwyyxx/tensorpack/blob/master/examples/Atari2600/DQN.py#L182>`_
            uses this callback to update target network.
        """
        self.setup_func = setup_func
        self.run_before = run_before
        self.run_as_trigger = run_as_trigger
        self.run_step = run_step

    def _setup_graph(self):
        self._op = self.setup_func()

    def _before_run(self, _):
        if self.run_step:
            return [self._op]

    def _before_train(self):
        if self.run_before:
            self._op.run()

    def _trigger(self):
        if self.run_as_trigger:
            self._op.run()


class RunUpdateOps(RunOp):
    """
    Run ops from the collection UPDATE_OPS every step
    """
    def __init__(self, collection=tf.GraphKeys.UPDATE_OPS):
        def f():
            ops = tf.get_collection(collection)
            if ops:
                logger.info("Applying UPDATE_OPS collection of {} ops.".format(len(ops)))
                return tf.group(*ops, name='update_ops')
            else:
                return tf.no_op(name='empty_update_ops')

        super(RunUpdateOps, self).__init__(
            f, run_before=False, run_as_trigger=False, run_step=True)
