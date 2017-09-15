#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: monitor.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import os
import numpy as np
import shutil
import time
import operator
from collections import defaultdict
import six
import json
import re

import tensorflow as tf
from ..utils import logger
from ..tfutils.summary import create_scalar_summary, create_image_summary
from .base import Callback

__all__ = ['TrainingMonitor', 'Monitors',
           'TFSummaryWriter', 'TFEventWriter', 'JSONWriter',
           'ScalarPrinter', 'SendMonitorData']


def image_to_nhwc(arr):
    if arr.ndim == 4:
        pass
    elif arr.ndim == 3:
        if arr.shape[-1] in [1, 3, 4]:
            arr = arr[np.newaxis, :]
        else:
            arr = arr[:, :, :, np.newaxis]
    elif arr.ndim == 2:
        arr = arr[np.newaxis, :, :, np.newaxis]
    else:
        raise ValueError("Array of shape {} is not an image!".format(arr.shape))
    return arr


class TrainingMonitor(Callback):
    """
    Monitor a training progress, by processing different types of
    summary/statistics from trainer.

    .. document private functions
    .. automethod:: _setup_graph
    """
    def setup_graph(self, trainer):
        self.trainer = trainer
        self._setup_graph()

    def _setup_graph(self):
        """ Override this method to setup the monitor."""
        pass

    def process_summary(self, summary):
        """
        Process a tf.Summary.
        """
        pass

    def process(self, name, val):
        """
        Process a key-value pair.
        """
        pass

    def process_scalar(self, name, val):
        """
        Args:
            val: a scalar
        """
        pass

    def process_image(self, name, val):
        """
        Args:
            val (np.ndarray): 4D (NHWC) numpy array of images in range [0,255].
                If channel is 3, assumed to be RGB.
        """
        pass

    def process_event(self, evt):
        """
        Args:
            evt (tf.Event): the most basic format acceptable by tensorboard.
                It could include Summary, RunMetadata, LogMessage, and more.
        """
        pass
    # TODO process other types


class NoOpMonitor(TrainingMonitor):
    pass


class Monitors(Callback):
    """
    Merge monitors together for trainer to use.

    In training, each trainer will create a :class:`Monitors` instance,
    and you can access it through `trainer.monitors`.
    You should use `trainer.monitors` for logging and it will dispatch your
    logs to each sub-monitor.
    """
    def __init__(self, monitors):
        self._scalar_history = ScalarHistory()
        self._monitors = monitors + [self._scalar_history]
        for m in self._monitors:
            assert isinstance(m, TrainingMonitor), m

    def _setup_graph(self):
        self._scalar_history.setup_graph(self.trainer)

    def _dispatch(self, func):
        for m in self._monitors:
            func(m)

    def put_summary(self, summary):
        """
        Put a `tf.Summary`.
        """
        if isinstance(summary, six.binary_type):
            summary = tf.Summary.FromString(summary)
        assert isinstance(summary, tf.Summary), type(summary)

        # TODO other types
        for val in summary.value:
            if val.WhichOneof('value') == 'simple_value':
                val.tag = re.sub('tower[0-9]+/', '', val.tag)   # TODO move to subclasses
                suffix = '-summary'  # tensorflow#6150, tensorboard#59
                if val.tag.endswith(suffix):
                    val.tag = val.tag[:-len(suffix)]
                self._dispatch(lambda m: m.process_scalar(val.tag, val.simple_value))

        self._dispatch(lambda m: m.process_summary(summary))

    def put_scalar(self, name, val):
        """
        Put a scalar.
        """
        if isinstance(val, np.floating):
            val = float(val)
        if isinstance(val, np.integer):
            val = int(val)
        self._dispatch(lambda m: m.process_scalar(name, val))
        s = create_scalar_summary(name, val)
        self._dispatch(lambda m: m.process_summary(s))

    def put_image(self, name, val):
        """
        Put an image.
        Args:
            name (str):
            val (np.ndarray): 2D, 3D (HWC) or 4D (NHWC) numpy array of images
                in range [0,255].  If channel is 3, assumed to be RGB.
        """
        assert isinstance(val, np.ndarray)
        arr = image_to_nhwc(val)
        self._dispatch(lambda m: m.process_image(name, arr))
        s = create_image_summary(name, arr)
        self._dispatch(lambda m: m.process_summary(s))

    def put_event(self, evt):
        """
        Put an tf.Event.
        `step` and `wall_time` fields of :class:`tf.Event` will be filled automatically.

        Args:
            evt (tf.Event):
        """
        evt.step = self.global_step
        evt.wall_time = time.time()
        self._dispatch(lambda m: m.process_event(evt))

    def get_latest(self, name):
        """
        Get latest scalar value of some data.
        """
        return self._scalar_history.get_latest(name)

    def get_history(self, name):
        """
        Get a history of the scalar value of some data.
        """
        return self._scalar_history.get_history(name)


class TFEventWriter(TrainingMonitor):
    """
    Write summaries to TensorFlow event file.
    """
    def __new__(cls):
        if logger.LOG_DIR:
            return super(TFEventWriter, cls).__new__(cls)
        else:
            logger.warn("logger directory was not set. Ignore TFEventWriter.")
            return NoOpMonitor()

    def _setup_graph(self):
        self._writer = tf.summary.FileWriter(logger.LOG_DIR, graph=tf.get_default_graph())

    def process_summary(self, summary):
        self._writer.add_summary(summary, self.global_step)

    def process_event(self, evt):
        self._writer.add_event(evt)

    def _trigger(self):     # flush every epoch
        self._writer.flush()

    def _after_train(self):
        self._writer.close()


def TFSummaryWriter(*args, **kwargs):
    logger.warn("TFSummaryWriter was renamed to TFEventWriter!")
    return TFEventWriter(*args, **kwargs)


class JSONWriter(TrainingMonitor):
    """
    Write all scalar data to a json file under ``logger.LOG_DIR``, grouped by their global step.
    This monitor also attemps to recover the epoch number during setup,
    if an existing json file is found at the same place.
    """

    FILENAME = 'stat.json'
    """
    The name of the json file.
    """

    def __new__(cls):
        if logger.LOG_DIR:
            return super(JSONWriter, cls).__new__(cls)
        else:
            logger.warn("logger directory was not set. Ignore JSONWriter.")
            return NoOpMonitor()

    def _setup_graph(self):
        self._dir = logger.LOG_DIR
        self._fname = os.path.join(self._dir, self.FILENAME)

        if os.path.isfile(self._fname):
            # TODO make a backup first?
            logger.info("Found existing JSON at {}, will append to it.".format(self._fname))
            with open(self._fname) as f:
                self._stats = json.load(f)
                assert isinstance(self._stats, list), type(self._stats)

            try:
                epoch = self._stats[-1]['epoch_num'] + 1
            except Exception:
                pass
            else:
                logger.info("Found training history from JSON, now starting from epoch number {}.".format(epoch))
                self.trainer.config.starting_epoch = epoch
        else:
            self._stats = []
        self._stat_now = {}

        self._last_gs = -1
        self._total = self.trainer.config.steps_per_epoch

    def _trigger_step(self):
        # will do this in trigger_epoch
        if self.local_step != self._total - 1:
            self._push()

    def _trigger_epoch(self):
        self._push()

    def process_scalar(self, name, val):
        self._stat_now[name] = val

    def _push(self):
        """ Note that this method is idempotent"""
        if len(self._stat_now):
            self._stat_now['epoch_num'] = self.epoch_num
            self._stat_now['global_step'] = self.global_step

            self._stats.append(self._stat_now)
            self._stat_now = {}
            self._write_stat()

    def _write_stat(self):
        tmp_filename = self._fname + '.tmp'
        try:
            with open(tmp_filename, 'w') as f:
                json.dump(self._stats, f)
            shutil.move(tmp_filename, self._fname)
        except IOError:  # disk error sometimes..
            logger.exception("Exception in JSONWriter._write_stat()!")


class ScalarPrinter(TrainingMonitor):
    """
    Print scalar data into terminal.
    """
    def __init__(self, enable_step=False, enable_epoch=True,
                 whitelist=None, blacklist=None):
        """
        Args:
            enable_step, enable_epoch (bool): whether to print the
                monitor data (if any) between steps or between epochs.
            whitelist (list[str] or None): A list of regex. Only names
                matching some regex will be allowed for printing.
                Defaults to match all names.
            blacklist (list[str] or None): A list of regex. Names matching
                any regex will not be printed. Defaults to match no names.
        """
        def compile_regex(rs):
            if rs is None:
                return None
            rs = set([r if isinstance(r, re.RegexObject) else re.compile(r) for r in rs])
            return rs

        self._whitelist = compile_regex(whitelist)
        if blacklist is None:
            blacklist = []
        self._blacklist = compile_regex(blacklist)

        self._enable_step = enable_step
        self._enable_epoch = enable_epoch

    def _setup_graph(self):
        self._dic = {}
        self._total = self.trainer.config.steps_per_epoch

    def _trigger_step(self):
        if self._enable_step:
            if self.local_step != self._total - 1:
                # not the last step
                self._print_stat()
            else:
                if not self._enable_epoch:
                    self._print_stat()
                # otherwise, will print them together

    def _trigger_epoch(self):
        if self._enable_epoch:
            self._print_stat()

    def process_scalar(self, name, val):
        self._dic[name] = float(val)

    def _print_stat(self):
        def match_regex_list(regexs, name):
            for r in regexs:
                if r.search(name) is not None:
                    return True
            return False

        for k, v in sorted(self._dic.items(), key=operator.itemgetter(0)):
            if self._whitelist is None or \
                    match_regex_list(self._whitelist, k):
                if not match_regex_list(self._blacklist, k):
                    logger.info('{}: {:.5g}'.format(k, v))
        self._dic = {}


class ScalarHistory(TrainingMonitor):
    """
    Only used by monitors internally.
    """
    def _setup_graph(self):
        self._dic = defaultdict(list)

    def process_scalar(self, name, val):
        self._dic[name].append(float(val))

    def get_latest(self, name):
        hist = self._dic[name]
        if len(hist) == 0:
            raise KeyError("Invalid key: {}".format(name))
        else:
            return hist[-1]

    def get_history(self, name):
        return self._dic[name]


class SendMonitorData(TrainingMonitor):
    """
    Execute a command with some specific scalar monitor data.
    This is useful for, e.g. building a custom statistics monitor.

    It will try to send once receiving all the stats
    """
    def __init__(self, command, names):
        """
        Args:
            command(str): a command to execute. Use format string with stat
                names as keys.
            names(list or str): data name(s) to use.

        Example:
            Send the stats to your phone through pushbullet:

            .. code-block:: python

                SendMonitorData('curl -u your_id: https://api.pushbullet.com/v2/pushes \\
                         -d type=note -d title="validation error" \\
                         -d body={validation_error} > /dev/null 2>&1',
                         'validation_error')
        """
        self.command = command
        if not isinstance(names, list):
            names = [names]
        self.names = names
        self.dic = {}

    def process_scalar(self, name, val):
        if name in self.names:
            self.dic[name] = val

    def _trigger_step(self):
        self._try_send()

    def _trigger_epoch(self):
        self._try_send()

    def _try_send(self):
        try:
            v = {k: self.dic[k] for k in self.names}
        except KeyError:
            return
        cmd = self.command.format(**v)
        ret = os.system(cmd)
        if ret != 0:
            logger.error("Command '{}' failed with ret={}!".format(cmd, ret))
        self.dic = {}
