# !/usr/bin/env python2
#  -*- coding: UTF-8 -*-
#  File: __init__.py
#  Author: Yuxin Wu <ppwwyyxx@gmail.com>

from pkgutil import walk_packages
import os
import time
import sys
from contextlib import contextmanager
import logger

def global_import(name):
    p = __import__(name, globals(), locals())
    lst = p.__all__ if '__all__' in dir(p) else dir(p)
    for k in lst:
        globals()[k] = p.__dict__[k]

global_import('naming')
global_import('callback')
global_import('validation_callback')


@contextmanager
def timed_operation(msg, log_start=False):
    if log_start:
        logger.info('start {} ...'.format(msg))
    start = time.time()
    yield
    logger.info('finished {}, time={:.2f}sec.'.format(
        msg, time.time() - start))
