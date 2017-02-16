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


@contextmanager
def timed_operation(msg, log_start=False):
    """
    Surround a context with a timer.

    Args:
        msg(str): the log to print.
        log_start(bool): whether to print also at the beginning.

    Example:
        .. code-block:: python

            with timed_operation('Good Stuff'):
                time.sleep(1)

        Will print:

        .. code-block:: python

            Good stuff finished, time:1sec.
    """
    if log_start:
        logger.info('Start {} ...'.format(msg))
    start = time.time()
    yield
    logger.info('{} finished, time:{:.4f}sec.'.format(
        msg, time.time() - start))


_TOTAL_TIMER_DATA = defaultdict(StatCounter)


@contextmanager
def total_timer(msg):
    """ A context which add the time spent inside to TotalTimer. """
    start = time.time()
    yield
    t = time.time() - start
    _TOTAL_TIMER_DATA[msg].feed(t)


def print_total_timer():
    """
    Print the content of the TotalTimer, if it's not empty. This function will automatically get
    called when program exits.
    """
    if len(_TOTAL_TIMER_DATA) == 0:
        return
    for k, v in six.iteritems(_TOTAL_TIMER_DATA):
        logger.info("Total Time: {} -> {:.2f} sec, {} times, {:.3g} sec/time".format(
            k, v.sum, v.count, v.average))


atexit.register(print_total_timer)


class IterSpeedCounter(object):
    """ Test how often some code gets reached.

    Example:
        Print the speed of the iteration every 100 times.

        .. code-block:: python

            speed = IterSpeedCounter(100)
            for k in range(1000):
                # do something
                speed()
    """

    def __init__(self, print_every, name=None):
        """
        Args:
            print_every(int): interval to print.
            name(str): name to used when print.
        """
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
