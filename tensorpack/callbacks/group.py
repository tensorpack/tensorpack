# -*- coding: UTF-8 -*-
# File: group.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import tensorflow as tf
from contextlib import contextmanager
from collections import defaultdict
import time
import traceback

from .base import Callback
from .stats import StatPrinter
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
    A container to hold all callbacks, and execute them in the right order
    (e.g. :class:`StatPrinter` will be executed at last).
    """

    def __init__(self, cbs):
        """
        Args:
            cbs(list): a list of :class:`Callback` instances.
        """
        # check type
        for cb in cbs:
            assert isinstance(cb, Callback), cb.__class__
        # move "StatPrinter" to the last
        # TODO don't need to manually move in the future.
        for idx, cb in enumerate(cbs):
            if isinstance(cb, StatPrinter):
                sp = cb
                cbs.remove(sp)
                cbs.append(sp)
                if idx != len(cbs) - 1:
                    logger.warn("StatPrinter should appear as the last element of callbacks! "
                                "This is now fixed automatically, but may not work in the future.")
                break

        self.cbs = cbs
        self._extra_fetches_cache = None

    def _setup_graph(self):
        with tf.name_scope(None):
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

    def _extra_fetches(self):
        if self._extra_fetches_cache is not None:
            return self._extra_fetches_cache
        # TODO use dispatch mechanism to avoid duplication
        self._cbid_to_fetchid = defaultdict(list)
        ret = []
        for idx, cb in enumerate(self.cbs):
            fetch = cb.extra_fetches()
            if len(fetch) == 0:
                continue
            for f in fetch:
                ret.append(f)
                self._cbid_to_fetchid[idx].append(len(ret)-1)
        self._extra_fetches_cache = ret
        return ret

    def _trigger_step(self, *args):
        for idx, cb in enumerate(self.cbs):
            fid = self._cbid_to_fetchid[idx]
            if len(fid) == 0:
                cb.trigger_step()
            else:
                data = [args[k] for k in fid]
                cb.trigger_step(*data)

    def _trigger_epoch(self):
        tm = CallbackTimeLogger()

        for cb in self.cbs:
            display_name = str(cb)
            with tm.timed_callback(display_name):
                cb.trigger_epoch()
        tm.log()

    def append(self, cb):
        assert isinstance(cb, Callback)
        self.cbs.append(cb)
