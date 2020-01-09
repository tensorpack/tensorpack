# -*- coding: utf-8 -*-
# File: prof.py


import multiprocessing as mp
import numpy as np
import os
import time
import tensorflow as tf
from six.moves import map, queue
import psutil

from ..tfutils.common import gpu_available_in_session
from ..utils import logger
from ..utils.timer import Timer
from ..utils.concurrency import ensure_proc_terminate, start_proc_mask_signal
from ..utils.gpu import get_num_gpu
from ..utils.nvml import NVMLContext
from .base import Callback

__all__ = ['GPUUtilizationTracker', 'GraphProfiler', 'PeakMemoryTracker',
           'GPUMemoryTracker', 'HostMemoryTracker', 'ThroughputTracker']


class GPUUtilizationTracker(Callback):
    """ Summarize the average GPU utilization within an epoch.

    It will start a process to obtain GPU utilization through NVML every second
    within the epoch (the trigger_epoch time was not included),
    and write average utilization to monitors.

    This callback creates a process, therefore it's not safe to be used with MPI.
    """

    _chief_only = False

    def __init__(self, devices=None):
        """
        Args:
            devices (list[int]): physical GPU ids to monitor. If None, will guess from the environment.
        """
        assert os.name != 'nt', "GPUUtilizationTracker does not support windows!"
        self._devices = devices
        self._enabled = True

    def _guess_devices(self):
        env = os.environ.get('CUDA_VISIBLE_DEVICES')
        if env is None:
            devices = list(range(get_num_gpu()))
            if len(devices) > 1:
                logger.warn("[GPUUtilizationTracker] Both devices and CUDA_VISIBLE_DEVICES are None! "
                            "Will monitor all {} visible GPUs!".format(len(devices)))
        else:
            if len(env):
                devices = list(map(int, env.split(',')))
            else:
                devices = []
        return devices

    def _setup_graph(self):
        # special heuristics for Horovod
        from ..train import HorovodTrainer
        if isinstance(self.trainer, HorovodTrainer):
            if self.trainer.mpi_enabled():
                logger.warn("GPUUtilizationTracker is disabled under MPI.")
                self._enabled = False
                return
            else:
                self._devices = [self.trainer.hvd.local_rank()]

        if self._devices is None:
            self._devices = self._guess_devices()
        assert len(self._devices), "[GPUUtilizationTracker] No GPU device given!"

        self._evt = mp.Event()
        self._stop_evt = mp.Event()
        self._queue = mp.Queue()
        self._proc = mp.Process(target=self.worker, args=(
            self._evt, self._queue, self._stop_evt, self._devices))
        ensure_proc_terminate(self._proc)
        start_proc_mask_signal(self._proc)

    def _before_train(self):
        assert gpu_available_in_session(), "[GPUUtilizationTracker] needs GPU!"

    def _before_epoch(self):
        if self._enabled:
            self._evt.set()

    def _after_epoch(self):
        if self._enabled:
            while self._evt.is_set():   # unlikely, unless the epoch is extremely fast
                pass
            self._evt.set()

    def _trigger_epoch(self):
        # Don't do this in after_epoch because
        # before,after_epoch are supposed to be extremely fast by design.
        if not self._enabled:
            return
        try:
            stats = self._queue.get(timeout=60)
        except queue.Empty:
            if self._proc.is_alive():
                raise RuntimeError("GPUUtilization.worker() is stuck. This is a bug.")
            else:
                raise RuntimeError("GPUUtilization.worker() process is killed unexpectedly.")

        if isinstance(stats, int) and stats == -1:
            from ..train.base import StopTraining
            raise StopTraining("GPUUtilizationTracker.worker has failed.")
        for idx, dev in enumerate(self._devices):
            self.trainer.monitors.put_scalar('GPUUtil/{}'.format(dev), stats[idx])

    def _after_train(self):
        if self._enabled:
            self._stop_evt.set()
            self._evt.set()
            self._proc.terminate()

    @staticmethod
    def worker(evt, rst_queue, stop_evt, devices):
        """
        Args:
            devices (list[int])
        """
        with NVMLContext() as ctx:
            devices = [ctx.device(i) for i in devices]
            while True:
                try:
                    evt.wait()  # start epoch
                    evt.clear()
                    if stop_evt.is_set():   # or on exit
                        return

                    stats = np.zeros((len(devices),), dtype='f4')
                    cnt = 0
                    while True:
                        time.sleep(1)

                        data = [d.utilization()['gpu'] for d in devices]
                        data = list(map(float, data))
                        stats += data
                        cnt += 1

                        if evt.is_set():    # stop epoch
                            if stop_evt.is_set():   # or on exit
                                return
                            evt.clear()
                            if cnt > 1:
                                # Ignore the last datapoint. Usually is zero, makes us underestimate the util.
                                stats -= data
                                cnt -= 1
                            rst_queue.put(stats / cnt)
                            break
                except Exception:
                    logger.exception("Exception in GPUUtilizationTracker.worker")
                    rst_queue.put(-1)
                    return


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
        from tensorflow.python.client import timeline
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


class GPUMemoryTracker(Callback):
    """
    Track peak memory used on each GPU device every epoch, by :mod:`tf.contrib.memory_stats`.
    The peak memory comes from the ``MaxBytesInUse`` op, which is the peak memory used
    in recent ``session.run`` calls.
    See https://github.com/tensorflow/tensorflow/pull/13107.
    """

    _chief_only = False

    def __init__(self, devices=(0,)):
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

    def _before_train(self):
        assert gpu_available_in_session(), "PeakMemoryTracker only supports GPU!"

    def _before_run(self, _):
        if self.local_step == self.trainer.steps_per_epoch - 1:
            return self._fetches
        return None

    def _after_run(self, _, rv):
        results = rv.results
        if results is not None:
            for mem, dev in zip(results, self._devices):
                self.trainer.monitors.put_scalar('PeakMemory(MB)' + dev, mem / 1e6)


PeakMemoryTracker = GPUMemoryTracker


class HostMemoryTracker(Callback):
    """
    Track free RAM on the host.

    When triggered, it writes the size of free RAM into monitors.
    """
    _chief_only = False

    def _setup_graph(self):
        logger.info("[HostMemoryTracker] Free RAM in setup_graph() is {:.2f} GB.".format(self._free_ram_gb()))

    def _before_train(self):
        logger.info("[HostMemoryTracker] Free RAM in before_train() is {:.2f} GB.".format(self._free_ram_gb()))

    def _trigger(self):
        ram_gb = self._free_ram_gb()
        self.trainer.monitors.put_scalar('HostFreeMemory (GB)', ram_gb)

    def _free_ram_gb(self):
        return psutil.virtual_memory().available / 1024**3


class ThroughputTracker(Callback):
    """
    This callback writes the training throughput (in terms of either steps/sec, or samples/sec)
    to the monitors everytime it is triggered.
    The throughput is computed based on the duration between the consecutive triggers.

    The time spent on callbacks after each epoch is excluded.
    """

    _chief_only = False

    def __init__(self, samples_per_step=None):
        """
        Args:
            samples_per_step (int or None): total number of samples processed in each step
                (i.e., your total batch size in each step).
                If not provided, this callback will record "steps/sec" instead of "samples/sec".
        """
        if samples_per_step is not None:
            samples_per_step = int(samples_per_step)
        self._samples_per_step = samples_per_step
        self._timer = Timer()
        self._timer.pause()

    # only include the time between before_epoch/after_epoch
    def _before_epoch(self):
        self._timer.resume()

    def _after_epoch(self):
        self._timer.pause()

    def _before_train(self):
        self._update_last()

    def _update_last(self):
        old_pause = self._timer.is_paused()
        self._timer.reset()
        if old_pause:
            self._timer.pause()
        self._last_step = self.global_step

    def _trigger(self):
        steps_per_sec = (self.global_step - self._last_step) / self._timer.seconds()
        self._update_last()

        if self._samples_per_step is None:
            self.trainer.monitors.put_scalar("Throughput (steps/sec)", steps_per_sec)
        else:
            self.trainer.monitors.put_scalar("Throughput (samples/sec)", steps_per_sec * self._samples_per_step)
