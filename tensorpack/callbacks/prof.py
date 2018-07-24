# -*- coding: utf-8 -*-
# File: prof.py


import os
import numpy as np
import multiprocessing as mp
import time
from six.moves import map
import tensorflow as tf
from tensorflow.python.client import timeline

from .base import Callback
from ..utils import logger
from ..utils.concurrency import ensure_proc_terminate, start_proc_mask_signal
from ..utils.gpu import get_num_gpu
from ..utils.nvml import NVMLContext

__all__ = ['GPUUtilizationTracker', 'GraphProfiler', 'PeakMemoryTracker']


class GPUUtilizationTracker(Callback):
    """ Summarize the average GPU utilization within an epoch.

    It will start a process to run `nvidia-smi` every second
    within the epoch (the trigger_epoch time was not included),
    and write average utilization to monitors.

    This callback creates a process, therefore it's not safe to be used with MPI.
    """

    _chief_only = False

    def __init__(self, devices=None):
        """
        Args:
            devices (list[int]): physical GPU ids. If None, will use CUDA_VISIBLE_DEVICES
        """
        assert os.name != 'nt', "GPUUtilizationTracker does not support windows!"
        if devices is None:
            env = os.environ.get('CUDA_VISIBLE_DEVICES')
            if env is None:
                self._devices = list(range(get_num_gpu()))
                logger.warn("[GPUUtilizationTracker] Both devices and CUDA_VISIBLE_DEVICES are None! "
                            "Will monitor all {} visible GPUs!".format(len(self._devices)))
            else:
                if len(env):
                    self._devices = list(map(int, env.split(',')))
                else:
                    self._devices = []
        else:
            self._devices = devices
        assert len(self._devices), "[GPUUtilizationTracker] No GPU device given!"

    def _before_train(self):
        self._evt = mp.Event()
        self._stop_evt = mp.Event()
        self._queue = mp.Queue()
        self._proc = mp.Process(target=self.worker, args=(
            self._evt, self._queue, self._stop_evt))
        ensure_proc_terminate(self._proc)
        start_proc_mask_signal(self._proc)

    def _before_epoch(self):
        self._evt.set()

    def _after_epoch(self):
        while self._evt.is_set():   # unlikely
            pass
        self._evt.set()

    def _trigger_epoch(self):
        # Don't do this in after_epoch because
        # before,after_epoch are supposed to be extremely fast by design.
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
            with NVMLContext() as ctx:
                while True:
                    time.sleep(1)

                    data = [ctx.device(i).utilization()['gpu'] for i in self._devices]
                    data = list(map(float, data))
                    stats += data
                    cnt += 1

                    if evt.is_set():    # stop epoch
                        if stop_evt.is_set():   # or on exit
                            return
                        evt.clear()
                        # Ignore the last datapoint. Usually is zero, makes us underestimate the util.
                        stats -= data
                        cnt -= 1
                        rst_queue.put(stats / cnt)
                        break


# Can add more features from tfprof
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/profiler/README.md

class GraphProfiler(Callback):
    """
    Enable profiling by installing session hooks,
    and write tracing files / events / metadata to ``logger.get_logger_dir()``.

    The tracing files can be loaded from ``chrome://tracing``.
    The metadata files can be processed by
    `tfprof command line utils
    <https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/profiler/README.md>`_.
    The event is viewable from tensorboard.

    Tips:

    Note that the profiling is by default enabled for every step and is expensive.
    You probably want to schedule it less frequently, e.g.:

    .. code-block:: none

        EnableCallbackIf(
            GraphProfiler(dump_tracing=True, dump_event=True),
            lambda self: self.trainer.global_step > 20 and self.trainer.global_step < 30)
    """
    def __init__(self, dump_metadata=False, dump_tracing=True, dump_event=False):
        """
        Args:
            dump_metadata(bool): Dump :class:`tf.RunMetadata` to be used with tfprof.
            dump_tracing(bool): Dump chrome tracing files.
            dump_event(bool): Dump to an event processed by FileWriter and
                will be shown in TensorBoard.
        """
        self._dir = logger.get_logger_dir()
        self._dump_meta = bool(dump_metadata)
        self._dump_tracing = bool(dump_tracing)
        self._dump_event = bool(dump_event)
        assert os.path.isdir(self._dir), self._dir

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
    Track peak memory used on each GPU device every epoch, by :mod:`tf.contrib.memory_stats`.
    The peak memory comes from the `MaxBytesInUse` op, which might span
    multiple session.run.
    See https://github.com/tensorflow/tensorflow/pull/13107.
    """

    _chief_only = False

    def __init__(self, devices=[0]):
        """
        Args:
            devices([int] or [str]): list of GPU devices to track memory on.
        """
        assert isinstance(devices, (list, tuple)), devices
        devices = ['/gpu:{}'.format(x) if isinstance(x, int) else x for x in devices]
        self._devices = devices

    def _setup_graph(self):
        from tensorflow.contrib.memory_stats import MaxBytesInUse
        ops = []
        for dev in self._devices:
            with tf.device(dev):
                ops.append(MaxBytesInUse())
        self._fetches = tf.train.SessionRunArgs(fetches=ops)

    def _before_run(self, _):
        if self.local_step == self.trainer.steps_per_epoch - 1:
            return self._fetches
        return None

    def _after_run(self, _, rv):
        results = rv.results
        if results is not None:
            for mem, dev in zip(results, self._devices):
                self.trainer.monitors.put_scalar('PeakMemory(MB)' + dev, mem / 1e6)
