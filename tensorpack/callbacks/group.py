# -*- coding: utf-8 -*-
# File: group.py


import traceback
from contextlib import contextmanager
from time import perf_counter as timer  # noqa
from ..compat import tfv1 as tf

from ..utils import logger
from ..utils.utils import humanize_time_delta
from .base import Callback
from .hooks import CallbackToHook

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
        s = timer()
        yield
        self.add(name, timer() - s)

    def log(self):

        """ log the time of some heavy callbacks """
        if self.tot < 3:
            return
        msgs = []
        for name, t in self.times:
            if t / self.tot > 0.3 and t > 1:
                msgs.append(name + ": " + humanize_time_delta(t))
        logger.info(
            "Callbacks took {:.3f} sec in total. {}".format(
                self.tot, '; '.join(msgs)))


class Callbacks(Callback):
    """
    A container to hold all callbacks, and trigger them iteratively.

    This is only used by the base trainer to run all the callbacks.
    Users do not need to use this class.
    """

    def __init__(self, cbs):
        """
        Args:
            cbs(list): a list of :class:`Callback` instances.
        """
        # check type
        for cb in cbs:
            assert isinstance(cb, Callback), cb.__class__
        self.cbs = cbs

    def _setup_graph(self):
        with tf.name_scope(None):   # clear the name scope
            for cb in self.cbs:
                cb.setup_graph(self.trainer)

    def _before_train(self):
        for cb in self.cbs:
            cb.before_train()

    def _after_train(self):
        for cb in self.cbs:
            # make sure callbacks are properly finalized
            try:
                cb.after_train()
            except Exception:
                traceback.print_exc()

    def get_hooks(self):
        return [CallbackToHook(cb) for cb in self.cbs]

    def trigger_step(self):
        for cb in self.cbs:
            cb.trigger_step()

    def _trigger_epoch(self):
        tm = CallbackTimeLogger()

        for cb in self.cbs:
            display_name = str(cb)
            with tm.timed_callback(display_name):
                cb.trigger_epoch()
        tm.log()

    def _before_epoch(self):
        for cb in self.cbs:
            cb.before_epoch()

    def _after_epoch(self):
        for cb in self.cbs:
            cb.after_epoch()
