#!/usr/bin/env python2
# -*- coding: utf-8 -*-
# File: summary.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import tensorflow as tf
import re
import os
import operator
import pickle

from .base import Callback, PeriodicCallback
from ..utils import *

__all__ = ['StatHolder', 'StatPrinter']

class StatHolder(object):
    def __init__(self, log_dir, print_tag=None):
        self.print_tag = None if print_tag is None else set(print_tag)
        self.stat_now = {}

        self.log_dir = log_dir
        self.filename = os.path.join(log_dir, 'stat.pkl')
        if os.path.isfile(self.filename):
            logger.info("Loading stats from {}...".format(self.filename))
            with open(self.filename) as f:
                self.stat_history = pickle.load(f)
        else:
            self.stat_history = []

    def add_stat(self, k, v):
        self.stat_now[k] = v

    def finalize(self):
        self._print_stat()
        self.stat_history.append(self.stat_now)
        self.stat_now = {}
        self._write_stat()

    def _print_stat(self):
        for k, v in sorted(self.stat_now.items(), key=operator.itemgetter(0)):
            if self.print_tag is None or k in self.print_tag:
                logger.info('{}: {:.4f}'.format(k, v))

    def _write_stat(self):
        tmp_filename = self.filename + '.tmp'
        with open(tmp_filename, 'wb') as f:
            pickle.dump(self.stat_history, f)
        os.rename(tmp_filename, self.filename)

class StatPrinter(Callback):
    def __init__(self, print_tag=None):
        """ print_tag : a list of regex to match scalar summary to print
            if None, will print all scalar tags
        """
        self.print_tag = print_tag

    def _before_train(self):
        logger.stat_holder = StatHolder(logger.LOG_DIR, self.print_tag)
