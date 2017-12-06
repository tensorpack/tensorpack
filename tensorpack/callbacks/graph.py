#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: graph.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

""" Graph related callbacks"""

import tensorflow as tf
import os
import numpy as np

from ..utils import logger
from .base import Callback
from six.moves import zip

__all__ = ['RunOp', 'RunUpdateOps', 'ProcessTensors', 'DumpTensors', 'DumpTensor']


class RunOp(Callback):
    """ Run an Op. """

    def __init__(self, op,
                 run_before=True, run_as_trigger=True,
                 run_step=False, verbose=False):
        """
        Args:
            op (tf.Operation or function): an Op, or a function that returns the Op in the graph.
                The function will be called later (in the `setup_graph` callback).
            run_before (bool): run the Op before training
            run_as_trigger (bool): run the Op on every trigger
            run_step (bool): run the Op every step (along with training)
            verbose (bool): pring logs when the op is run.

        Examples:
            The `DQN Example
            <https://github.com/ppwwyyxx/tensorpack/blob/master/examples/DeepQNetwork/>`_
            uses this callback to update target network.
        """
        if not callable(op):
            self.setup_func = lambda: op  # noqa
        else:
            self.setup_func = op
        self.run_before = run_before
        self.run_as_trigger = run_as_trigger
        self.run_step = run_step
        self.verbose = verbose

    def _setup_graph(self):
        self._op = self.setup_func()
        if self.run_step:
            self._fetch = tf.train.SessionRunArgs(fetches=self._op)

    def _before_train(self):
        if self.run_before:
            self._print()
            self._op.run()

    def _trigger(self):
        if self.run_as_trigger:
            self._print()
            self._op.run()

    def _before_run(self, _):
        if self.run_step:
            self._print()
            return self._fetch

    def _print(self):
        if self.verbose:
            logger.info("Running Op {} ...".format(self._op.name))


class RunUpdateOps(RunOp):
    """
    Run ops from the collection UPDATE_OPS every step
    """

    _chief_only = False

    def __init__(self, collection=tf.GraphKeys.UPDATE_OPS):
        """
        Args:
            collection (str): collection of ops to run. Defaults to ``tf.GraphKeys.UPDATE_OPS``
        """
        name = 'UPDATE_OPS' if collection == tf.GraphKeys.UPDATE_OPS else collection

        def f():
            ops = tf.get_collection(collection)
            if ops:
                logger.info("Applying collection {} of {} ops.".format(name, len(ops)))
                return tf.group(*ops, name='update_ops')
            else:
                return tf.no_op(name='empty_update_ops')

        super(RunUpdateOps, self).__init__(
            f, run_before=False, run_as_trigger=False, run_step=True)


class ProcessTensors(Callback):
    """
    Fetch extra tensors **along with** each training step,
    and call some function over the values.
    You can use it to print tensors, save tensors to file, etc.

    Examples:

    .. code-block:: python

        ProcessTensors(['mycost1', 'mycost2'], lambda c1, c2: print(c1, c2, c1 + c2))
    """
    def __init__(self, names, fn):
        """
        Args:
            names (list[str]): names of tensors
            fn: a function taking all requested tensors as input
        """
        assert isinstance(names, (list, tuple)), names
        self._names = names
        self._fn = fn

    def _setup_graph(self):
        tensors = self.get_tensors_maybe_in_tower(self._names)
        self._fetch = tf.train.SessionRunArgs(fetches=tensors)

    def _before_run(self, _):
        return self._fetch

    def _after_run(self, _, rv):
        results = rv.results
        self._fn(*results)


class DumpTensors(ProcessTensors):
    """
    Dump some tensors to a file.
    Every step this callback fetches tensors and write them to a npz file
    under ``logger.get_logger_dir``.
    The dump can be loaded by ``dict(np.load(filename).items())``.
    """
    def __init__(self, names):
        """
        Args:
            names (list[str]): names of tensors
        """
        assert isinstance(names, (list, tuple)), names
        self._names = names
        dir = logger.get_logger_dir()

        def fn(*args):
            dic = {}
            for name, val in zip(self._names, args):
                dic[name] = val
            fname = os.path.join(
                dir, 'DumpTensor-{}.npz'.format(self.global_step))
            np.savez(fname, **dic)
        super(DumpTensors, self).__init__(names, fn)


DumpTensor = DumpTensors
