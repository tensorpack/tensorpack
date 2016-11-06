# -*- coding: UTF-8 -*-
# File: group.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import tensorflow as tf
from contextlib import contextmanager
import time

from .base import Callback
from .stat import StatPrinter
from ..utils import logger

__all__ = ['Callbacks']

class CallbackTimeLogger(object):
    def __init__(self):
        self.times = []
        self.tot = 0

    def add(self, name, time):
        self.tot += time
        self.times.append((name, time))

    @contextmanager
    def timed_callback(self, name):
        s = time.time()
        yield
        self.add(name, time.time() - s)

    def log(self):
        """ log the time of some heavy callbacks """
        if self.tot < 3:
            return
        msgs = []
        for name, t in self.times:
            if t / self.tot > 0.3 and t > 1:
                msgs.append("{}: {:.3f}sec".format(name, t))
        logger.info(
            "Callbacks took {:.3f} sec in total. {}".format(
                self.tot, '; '.join(msgs)))

class Callbacks(Callback):
    """
    A container to hold all callbacks, and execute them in the right order and proper session.
    """
    def __init__(self, cbs):
        """
        :param cbs: a list of `Callbacks`
        """
        # check type
        for cb in cbs:
            assert isinstance(cb, Callback), cb.__class__
        # move "StatPrinter" to the last
        for cb in cbs:
            if isinstance(cb, StatPrinter):
                sp = cb
                cbs.remove(sp)
                cbs.append(sp)
                break
        else:
            raise ValueError("Callbacks must contain StatPrinter for stat and writer to work properly!")

        self.cbs = cbs

    def _setup_graph(self):
        with tf.name_scope(None):
            for cb in self.cbs:
                cb.setup_graph(self.trainer)

    def _before_train(self):
        for cb in self.cbs:
            cb.before_train()

    def _after_train(self):
        for cb in self.cbs:
            cb.after_train()

    def trigger_step(self):
        for cb in self.cbs:
            cb.trigger_step()

    def _trigger_epoch(self):
        tm = CallbackTimeLogger()

        test_sess_restored = False
        for cb in self.cbs:
            display_name = str(cb)
            with tm.timed_callback(display_name):
                cb.trigger_epoch()
        tm.log()

    def append(self, cb):
        assert isinstance(cb, Callback)
        self.cbs.append(cb)
