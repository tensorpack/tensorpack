# -*- coding: utf-8 -*-
# File: stats.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import os

from .base import Triggerable
from ..utils import logger
from ..utils.develop import log_deprecated

__all__ = ['StatPrinter', 'SendStat']


class StatPrinter(Triggerable):
    def __init__(self, print_tag=None):
        log_deprecated("StatPrinter",
                       "No need to add StatPrinter to callbacks anymore!",
                       "2017-03-26")


# TODO make it into monitor?
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

    def _trigger(self):
        M = self.trainer.monitors
        v = {k: M.get_latest(k) for k in self.stats}
        cmd = self.command.format(**v)
        ret = os.system(cmd)
        if ret != 0:
            logger.error("Command {} failed with ret={}!".format(cmd, ret))
