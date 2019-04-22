# -*- coding: utf-8 -*-
# File: misc.py


import numpy as np
import os
import time
from collections import deque

from ..utils import logger
from ..utils.utils import humanize_time_delta
from .base import Callback

__all__ = ['SendStat', 'InjectShell', 'EstimatedTimeLeft']


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


class InjectShell(Callback):
    """
    Allow users to create a specific file as a signal to pause
    and iteratively debug the training.
    Once the :meth:`trigger` method is called, it detects whether the file exists, and opens an
    IPython/pdb shell if yes.
    In the shell, ``self`` is this callback, ``self.trainer`` is the trainer, and
    from that you can access everything else.

    Example:

    .. code-block:: none

        callbacks=[InjectShell('/path/to/pause-training.tmp'), ...]

        # the following command will pause the training and start a shell when the epoch finishes:
        $ touch /path/to/pause-training.tmp

    """

    def __init__(self, file='INJECT_SHELL.tmp', shell='ipython'):
        """
        Args:
           file (str): if this file exists, will open a shell.
           shell (str): one of 'ipython', 'pdb'
        """
        self._file = file
        assert shell in ['ipython', 'pdb']
        self._shell = shell
        logger.info("Create a file '{}' to open {} shell.".format(file, shell))

    def _trigger(self):
        if os.path.isfile(self._file):
            logger.info("File {} exists, entering shell.".format(self._file))
            self._inject()

    def _inject(self):
        trainer = self.trainer   # noqa
        if self._shell == 'ipython':
            import IPython as IP    # noqa
            IP.embed()
        elif self._shell == 'pdb':
            import pdb   # noqa
            pdb.set_trace()

    def _after_train(self):
        if os.path.isfile(self._file):
            os.unlink(self._file)


class EstimatedTimeLeft(Callback):
    """
    Estimate the time left until completion of training.
    """
    def __init__(self, last_k_epochs=5, median=True):
        """
        Args:
            last_k_epochs (int): Use the time spent on last k epochs to estimate total time left.
            median (bool): Use the mean or median time spent on last k epochs.
        """
        self._times = deque(maxlen=last_k_epochs)
        self._median = median

    def _before_train(self):
        self._max_epoch = self.trainer.max_epoch
        self._last_time = time.time()

    def _trigger_epoch(self):
        duration = time.time() - self._last_time
        self._last_time = time.time()
        self._times.append(duration)

        epoch_time = np.median(self._times) if self._median else np.mean(self._times)
        time_left = (self._max_epoch - self.epoch_num) * epoch_time
        if time_left > 0:
            logger.info("Estimated Time Left: " + humanize_time_delta(time_left))
