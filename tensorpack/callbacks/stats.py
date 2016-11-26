# -*- coding: utf-8 -*-
# File: stat.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import tensorflow as tf
import re, os
import operator
import json

from .base import Callback
from ..utils import logger
from ..tfutils.common import get_global_step

__all__ = ['StatHolder', 'StatPrinter', 'SendStat']

class StatHolder(object):
    """
    A holder to keep all statistics aside from tensorflow events.
    """
    def __init__(self, log_dir):
        """
        :param log_dir: directory to save the stats.
        """
        self.set_print_tag([])
        self.blacklist_tag = set()
        self.stat_now = {}

        self.log_dir = log_dir
        self.filename = os.path.join(log_dir, 'stat.json')
        if os.path.isfile(self.filename):
            logger.info("Found stats at {}, will append to it.".format(self.filename))
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

    def add_blacklist_tag(self, blacklist_tag):
        """ Disable printing for some tags """
        self.blacklist_tag |= set(blacklist_tag)

    def get_stat_now(self, key):
        """
        Return the value of a stat in the current epoch.
        """
        return self.stat_now[key]

    def get_stat_history(self, key):
        ret = []
        for h in self.stat_history:
            v = h.get(key, None)
            if v is not None: ret.append(v)
        v = self.stat_now.get(key, None)
        if v is not None: ret.append(v)
        return ret

    def finalize(self):
        """
        Called after finishing adding stats for this epoch. Will print and write stats to disk.
        """
        self._print_stat()
        self.stat_history.append(self.stat_now)
        self.stat_now = {}
        self._write_stat()

    def _print_stat(self):
        for k, v in sorted(self.stat_now.items(), key=operator.itemgetter(0)):
            if self.print_tag is None or k in self.print_tag:
                if k not in self.blacklist_tag:
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
        self._stat_holder = self.trainer.stat_holder
        self._stat_holder.set_print_tag(self.print_tag)
        self._stat_holder.add_blacklist_tag(['global_step', 'epoch_num'])

        # just try to add this stat earlier so SendStat can use
        self._stat_holder.add_stat('epoch_num', self.epoch_num + 1)

    def _trigger_epoch(self):
        # by default, add this two stat
        self._stat_holder.add_stat('global_step', get_global_step())
        self._stat_holder.finalize()
        self._stat_holder.add_stat('epoch_num', self.epoch_num + 1)

class SendStat(Callback):
    """
    Execute a command with some specific stats.
    For example, send the stats to your phone through pushbullet:

        SendStat('curl -u your_id: https://api.pushbullet.com/v2/pushes \
            -d type=note -d title="validation error" \
            -d body={validation_error} > /dev/null 2>&1',
                'validation_error')
    """
    def __init__(self, command, stats):
        self.command = command
        if not isinstance(stats, list):
            stats = [stats]
        self.stats = stats

    def _trigger_epoch(self):
        holder = self.trainer.stat_holder
        v = {k: holder.get_stat_now(k) for k in self.stats}
        cmd = self.command.format(**v)
        ret = os.system(cmd)
        if ret != 0:
            logger.error("Command {} failed with ret={}!".format(cmd, ret))
