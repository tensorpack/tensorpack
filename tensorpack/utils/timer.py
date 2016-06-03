#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: timer.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>


from contextlib import contextmanager
import time
from collections import defaultdict
import six

from .stat import StatCounter
from . import logger

__all__ = ['total_timer', 'timed_operation', 'print_total_timer']

@contextmanager
def timed_operation(msg, log_start=False):
    if log_start:
        logger.info('start {} ...'.format(msg))
    start = time.time()
    yield
    logger.info('{} finished, time={:.2f}sec.'.format(
        msg, time.time() - start))

_TOTAL_TIMER_DATA = defaultdict(StatCounter)

@contextmanager
def total_timer(msg):
    start = time.time()
    yield
    t = time.time() - start
    _TOTAL_TIMER_DATA[msg].feed(t)

def print_total_timer():
    for k, v in six.iteritems(_TOTAL_TIMER_DATA):
        logger.info("Total Time: {} -> {} sec".format(k, v.sum))
