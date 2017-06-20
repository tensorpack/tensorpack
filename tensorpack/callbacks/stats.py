# -*- coding: utf-8 -*-
# File: stats.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import os
import numpy as np
import multiprocessing as mp
import time
from six.moves import map

from .base import Callback
from ..utils import logger
from ..utils.concurrency import ensure_proc_terminate, subproc_call

__all__ = ['SendStat', 'GPUUtilizationTracker']


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


class GPUUtilizationTracker(Callback):
    """ Summarize the average GPU utilization within an epoch"""

    def __init__(self, devices=None):
        """
        Args:
            devices (list[int]): physical GPU ids. If None, will use CUDA_VISIBLE_DEVICES
        """
        if devices is None:
            env = os.environ.get('CUDA_VISIBLE_DEVICES')
            assert env is not None, "[GPUUtilizationTracker] Both devices and CUDA_VISIBLE_DEVICES are None!"
            self._devices = env.split(',')
        else:
            self._devices = list(map(str, devices))
        assert len(self._devices), "[GPUUtilizationTracker] No GPU device given!"

        self._command = "nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i " + \
            ','.join(self._devices)
        _, ret = subproc_call(self._command)
        assert ret == 0, "Cannot fetch GPU utilization!"

    def _before_train(self):
        self._evt = mp.Event()
        self._stop_evt = mp.Event()
        self._queue = mp.Queue()
        self._proc = mp.Process(target=self.worker, args=(
            self._evt, self._queue, self._stop_evt))
        ensure_proc_terminate(self._proc)
        self._proc.start()

    def _before_epoch(self):
        self._evt.set()

    def _after_epoch(self):
        while self._evt.is_set():   # unlikely
            pass
        self._evt.set()
        stats = self._queue.get()
        for idx, dev in enumerate(self._devices):
            self.trainer.monitors.put_scalar('GPU{}-Util'.format(dev), stats[idx])

    def _after_train(self):
        self._stop_evt.set()
        self._evt.set()
        self._proc.join()

    def worker(self, evt, rst_queue, stop_evt):
        while True:
            evt.wait()  # start epoch
            evt.clear()
            if stop_evt.is_set():   # or on exit
                return

            stats = np.zeros((len(self._devices),), dtype='f4')
            cnt = 0
            while True:
                time.sleep(1)
                output, retv = subproc_call(self._command)
                assert retv == 0, "Cannot fetch GPU Utilization!"
                data = list(map(float, output.strip().split(b'\n')))
                stats += data
                cnt += 1

                if evt.is_set():    # stop epoch
                    if stop_evt.is_set():   # or on exit
                        return
                    evt.clear()
                    rst_queue.put(stats / cnt)
                    break
