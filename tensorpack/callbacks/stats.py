# -*- coding: utf-8 -*-
# File: stats.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import os
import operator
import json

from .base import Triggerable
from ..utils import logger

__all__ = ['StatHolder', 'StatPrinter', 'SendStat']


class StatHolder(object):
    """
    A holder to keep all statistics aside from tensorflow events.
    """

    def __init__(self, log_dir):
        """
        Args:
            log_dir(str): directory to save the stats.
        """
        self.set_print_tag([])
        self.blacklist_tag = set()
        self.stat_now = {}

        self.log_dir = log_dir
        self.filename = os.path.join(log_dir, 'stat.json')
        if os.path.isfile(self.filename):
            # TODO make a backup first?
            logger.info("Found stats at {}, will append to it.".format(self.filename))
            with open(self.filename) as f:
                self.stat_history = json.load(f)
        else:
            self.stat_history = []

        # global step of the current list of stat
        self._current_gs = -1

    def add_stat(self, k, v, global_step, epoch_num):
        """
        Add a stat.
        """
        if global_step != self._current_gs:
            self._push()
            self._current_gs = global_step
            self.stat_now['epoch_num'] = epoch_num
            self.stat_now['global_step'] = global_step
        self.stat_now[k] = float(v)

    def set_print_tag(self, print_tag):
        """
        Set name of stats to print.

        Args:
            print_tag: a collection of string.
        """
        self.print_tag = None if print_tag is None else set(print_tag)

    def add_blacklist_tag(self, blacklist_tag):
        """ Disable printing for some tags

        Args:
            blacklist_tag: a collection of string.
        """
        self.blacklist_tag |= set(blacklist_tag)

    def get_stat_now(self, key):
        """
        Return the value of a stat in the current epoch.

        Raises:
            KeyError if the key hasn't been added in this epoch.
        """
        return self.stat_now[key]

    def get_stat_history(self, key):
        """
        Returns:
            list: all history of a stat. Empty if there is not history of this name.
        """
        ret = []
        for h in self.stat_history:
            v = h.get(key, None)
            if v is not None:
                ret.append(v)
        v = self.stat_now.get(key, None)
        if v is not None:
            ret.append(v)
        return ret

    def finalize(self):
        """
        Print and write stats to disk.
        This method is idempotent.
        """
        self._print_stat()
        self._push()

    def _push(self):
        """ Note that this method is idempotent"""
        if len(self.stat_now):
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
        except IOError:  # disk error sometimes..
            logger.exception("Exception in StatHolder.finalize()!")


class StatPrinter(Triggerable):
    """
    A callback to control what stats to print. Enable by default to print
    everything in trainer.stat_holder.
    """

    def __init__(self, print_tag=None):
        """
        Args:
            print_tag: a list of stat names to print.
                If None, will print all scalar tags.
        """
        self.print_tag = print_tag

    def _before_train(self):
        self._stat_holder = self.trainer.stat_holder
        self._stat_holder.set_print_tag(self.print_tag)
        self._stat_holder.add_blacklist_tag(['global_step', 'epoch_num'])

    def _trigger(self):
        self._stat_holder.finalize()


class SendStat(Triggerable):
    """
    Execute a command with some specific stats.
    This is useful for, e.g. building a custom statistics monitor.
    """
    def __init__(self, command, stats):
        """
        Args:
            command(str): a command to execute. Use format string with stat
                names as keys.
            stats(list or str): stat name(s) to use.

        Example:
            Send the stats to your phone through pushbullet:

            .. code-block:: python

                SendStat('curl -u your_id: https://api.pushbullet.com/v2/pushes \\
                         -d type=note -d title="validation error" \\
                         -d body={validation_error} > /dev/null 2>&1',
                         'validation_error')
        """
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
