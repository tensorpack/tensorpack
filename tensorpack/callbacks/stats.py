# -*- coding: utf-8 -*-
# File: stats.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import os

from .base import Callback
from ..utils import logger

__all__ = ['SendStat']


class SendStat(Callback):
    """ An equivalent of :class:`SendMonitorData`, but as a normal callback. """
    def __init__(self, command, names):
        self.command = command
        if not isinstance(names, list):
            names = [names]
        self.names = names

    def _trigger(self):
        M = self.trainer.monitors
        v = {k: M.get_latest(k) for k in self.names}
        cmd = self.command.format(**v)
        ret = os.system(cmd)
        if ret != 0:
            logger.error("Command {} failed with ret={}!".format(cmd, ret))
