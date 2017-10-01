#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: prof.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import os
import numpy as np
import multiprocessing as mp
import time
from six.moves import map
import tensorflow as tf
from tensorflow.python.client import timeline

from .base import Callback
from ..utils import logger
from ..utils.concurrency import ensure_proc_terminate, subproc_call
from ..utils.gpu import get_nr_gpu

__all__ = ['GPUUtilizationTracker', 'GraphProfiler', 'PeakMemoryTracker']


class GPUUtilizationTracker(Callback):
    """ Summarize the average GPU utilization within an epoch.

    It will start a process to run `nvidia-smi` every second
    within the epoch (the trigger_epoch time was not included),
    and write average utilization to monitors.
    """

    def __init__(self, devices=None):
        """
        Args:
            devices (list[int]): physical GPU ids. If None, will use CUDA_VISIBLE_DEVICES
        """
        if devices is None:
            env = os.environ.get('CUDA_VISIBLE_DEVICES')
            if env is None:
                logger.warn("[GPUUtilizationTracker] Both devices and CUDA_VISIBLE_DEVICES are None! "
                            "Will monitor all visible GPUs!")
                self._devices = list(map(str, range(get_nr_gpu())))
            else:
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
            self.trainer.monitors.put_scalar('GPUUtil/{}'.format(dev), stats[idx])

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


# Can add more features from tfprof
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/profiler/README.md

class GraphProfiler(Callback):
    """
    Enable profiling by installing session hooks,
    and write metadata or tracing files to ``logger.LOG_DIR``.

    The tracing files can be loaded from ``chrome://tracing``.
    The metadata files can be processed by
    `tfprof command line utils
    <https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/profiler/README.md>`_.
    The event is viewable from tensorboard.

    Note that the profiling is enabled for every step.
    You probably want to schedule it less frequently by
    :class:`PeriodicRunHooks`.
    """
    def __init__(self, dump_metadata=False, dump_tracing=True, dump_event=False):
        """
        Args:
            dump_metadata(bool): Dump :class:`tf.RunMetadata` to be used with tfprof.
            dump_tracing(bool): Dump chrome tracing files.
            dump_event(bool): Dump to an event processed by FileWriter and
                will be shown in TensorBoard.
        """
        self._dir = logger.LOG_DIR
        self._dump_meta = bool(dump_metadata)
        self._dump_tracing = bool(dump_tracing)
        self._dump_event = bool(dump_event)
        assert os.path.isdir(self._dir)

    def _before_run(self, _):
        opt = tf.RunOptions()
        opt.trace_level = tf.RunOptions.FULL_TRACE
        return tf.train.SessionRunArgs(fetches=None, options=opt)

    def _after_run(self, _, run_values):
        meta = run_values.run_metadata
        if self._dump_meta:
            self._write_meta(meta)
        if self._dump_tracing:
            self._write_tracing(meta)
        if self._dump_event:
            self._write_event(meta)

    def _write_meta(self, metadata):
        fname = os.path.join(
            self._dir, 'runmetadata-{}.pb'.format(self.global_step))
        with open(fname, 'wb') as f:
            f.write(metadata.SerializeToString())

    def _write_tracing(self, metadata):
        tl = timeline.Timeline(step_stats=metadata.step_stats)
        fname = os.path.join(
            self._dir, 'chrome-trace-{}.json'.format(self.global_step))
        with open(fname, 'w') as f:
            f.write(tl.generate_chrome_trace_format(
                show_dataflow=True, show_memory=True))

    def _write_event(self, metadata):
        evt = tf.Event()
        evt.tagged_run_metadata.tag = 'trace-{}'.format(self.global_step)
        evt.tagged_run_metadata.run_metadata = metadata.SerializeToString()
        self.trainer.monitors.put_event(evt)


class PeakMemoryTracker(Callback):
    """
    Track peak memory in each session run, by
    :module:`tf.contrib.memory_stats`.
    It can only be used for GPUs.
    """
    def __init__(self, devices=['/gpu:0']):
        """
        Args:
            devices([str]): list of devices to track memory on.
        """
        self._devices = devices

    def _setup_graph(self):
        from tensorflow.contrib.memory_stats import MaxBytesInUse
        ops = []
        for dev in self._devices:
            with tf.device(dev):
                ops.append(MaxBytesInUse())
        self._fetches = tf.train.SessionRunArgs(fetches=ops)

    def _before_run(self, _):
        return self._fetches

    def _after_run(self, _, rv):
        results = rv.results
        for mem, dev in zip(results, self._devices):
            self.trainer.monitors.put_scalar('PeakMemory(MB)' + dev, mem / 1e6)
