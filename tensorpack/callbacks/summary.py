# -*- coding: utf-8 -*-
# File: summary.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import tensorflow as tf
import re
import os
import operator
import json

from .base import Callback
from ..utils import *

__all__ = ['StatHolder', 'StatPrinter']

class StatHolder(object):
    """
    A holder to keep all statistics aside from tensorflow events.
    """
    def __init__(self, log_dir):
        """
        :param log_dir: directory to save the stats.
        """
        self.set_print_tag([])
        self.stat_now = {}

        self.log_dir = log_dir
        self.filename = os.path.join(log_dir, 'stat.json')
        if os.path.isfile(self.filename):
            logger.info("Loading stats from {}...".format(self.filename))
            with open(self.filename) as f:
                self.stat_history = json.load(f)
        else:
            self.stat_history = []

    def add_stat(self, k, v):
        """
        Add a stat.
        :param k: name
        :param v: value
        """
        self.stat_now[k] = float(v)

    def set_print_tag(self, print_tag):
        """
        Set name of stats to print.
        """
        self.print_tag = None if print_tag is None else set(print_tag)

    def get_stat_now(self, key):
        """
        Return the value of a stat in the current epoch.
        """
        return self.stat_now[key]

    def finalize(self):
        """
        Called after finishing adding stats. Will print and write stats to disk.
        """
        self._print_stat()
        self.stat_history.append(self.stat_now)
        self.stat_now = {}
        self._write_stat()

    def _print_stat(self):
        for k, v in sorted(self.stat_now.items(), key=operator.itemgetter(0)):
            if self.print_tag is None or k in self.print_tag:
                logger.info('{}: {:.5g}'.format(k, v))

    def _write_stat(self):
        tmp_filename = self.filename + '.tmp'
        try:
            with open(tmp_filename, 'w') as f:
                json.dump(self.stat_history, f)
            os.rename(tmp_filename, self.filename)
        except IOError: # disk error sometimes..
            logger.exception("Exception in StatHolder.finalize()!")

class StatPrinter(Callback):
    """
    Control what stats to print.
    """
    def __init__(self, print_tag=None):
        """
        :param print_tag: a list of regex to match scalar summary to print.
            If None, will print all scalar tags
        """
        self.print_tag = print_tag

    def _before_train(self):
        self.trainer.stat_holder.set_print_tag(self.print_tag)
