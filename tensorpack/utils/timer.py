#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: timer.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>


from contextlib import contextmanager
import time
from collections import defaultdict
import six
import atexit

from .stats import StatCounter
from . import logger

__all__ = ['total_timer', 'timed_operation',
           'print_total_timer', 'IterSpeedCounter']


class IterSpeedCounter(object):
    """ To count how often some code gets reached"""

    def __init__(self, print_every, name=None):
        self.cnt = 0
        self.print_every = int(print_every)
        self.name = name if name else 'IterSpeed'

    def reset(self):
        self.start = time.time()

    def __call__(self):
        if self.cnt == 0:
            self.reset()
        self.cnt += 1
        if self.cnt % self.print_every != 0:
            return
        t = time.time() - self.start
        logger.info("{}: {:.2f} sec, {} times, {:.3g} sec/time".format(
            self.name, t, self.cnt, t / self.cnt))


@contextmanager
def timed_operation(msg, log_start=False):
    if log_start:
        logger.info('Start {} ...'.format(msg))
    start = time.time()
    yield
    logger.info('{} finished, time:{:.2f}sec.'.format(
        msg, time.time() - start))

_TOTAL_TIMER_DATA = defaultdict(StatCounter)


@contextmanager
def total_timer(msg):
    start = time.time()
    yield
    t = time.time() - start
    _TOTAL_TIMER_DATA[msg].feed(t)


def print_total_timer():
    if len(_TOTAL_TIMER_DATA) == 0:
        return
    for k, v in six.iteritems(_TOTAL_TIMER_DATA):
        logger.info("Total Time: {} -> {:.2f} sec, {} times, {:.3g} sec/time".format(
            k, v.sum, v.count, v.average))

atexit.register(print_total_timer)
