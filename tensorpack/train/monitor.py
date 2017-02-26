#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: monitor.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import os
import operator
from collections import defaultdict
import six
import json
import re

import tensorflow as tf
from ..utils import logger

__all__ = ['TrainingMonitor', 'Monitors',
           'TFSummaryWriter', 'JSONWriter', 'ScalarPrinter']


class TrainingMonitor(object):
    """
    Monitor a training progress, by processing different types of
    summary/statistics from trainer.

    .. document private functions
    .. automethod:: _setup
    """
    def setup(self, trainer):
        self._trainer = trainer
        self._setup()

    def _setup(self):
        """ Override this method to setup the monitor."""
        pass

    def put_summary(self, summary):
        """
        Process a tf.Summary.
        """
        pass

    def put(self, name, val):
        """
        Process a key-value pair.
        """
        pass

    def put_scalar(self, name, val):
        self.put(name, val)

    # TODO put other types

    def flush(self):
        pass

    def close(self):
        pass


class NoOpMonitor(TrainingMonitor):
    pass


class Monitors(TrainingMonitor):
    """
    Merge monitors together for trainer to use.
    """
    def __init__(self, monitors):
        # TODO filter by names
        self._scalar_history = ScalarHistory()
        self._monitors = monitors + [self._scalar_history]

    def setup(self, trainer):
        for m in self._monitors:
            m.setup(trainer)

    def flush(self):
        for m in self._monitors:
            m.flush()

    def close(self):
        for m in self._monitors:
            m.close()

    def _dispatch_put_summary(self, summary):
        for m in self._monitors:
            m.put_summary(summary)

    def _dispatch_put_scalar(self, name, val):
        for m in self._monitors:
            m.put_scalar(name, val)

    def put_summary(self, summary):
        if isinstance(summary, six.binary_type):
            summary = tf.Summary.FromString(summary)
        assert isinstance(summary, tf.Summary), type(summary)

        self._dispatch_put_summary(summary)

        # TODO other types
        for val in summary.value:
            if val.WhichOneof('value') == 'simple_value':
                val.tag = re.sub('tower[p0-9]+/', '', val.tag)   # TODO move to subclasses
                suffix = '-summary'  # issue#6150
                if val.tag.endswith(suffix):
                    val.tag = val.tag[:-len(suffix)]
                self._dispatch_put_scalar(val.tag, val.simple_value)

    def put(self, name, val):
        val = float(val)    # TODO only support scalar for now
        self.put_scalar(name, val)

    def put_scalar(self, name, val):
        self._dispatch_put_scalar(name, val)
        s = tf.Summary()
        s.value.add(tag=name, simple_value=val)
        self._dispatch_put_summary(s)

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


class TFSummaryWriter(TrainingMonitor):
    """
    Write summaries to TensorFlow event file.
    """
    def __new__(cls):
        if logger.LOG_DIR:
            return super(TFSummaryWriter, cls).__new__(cls)
        else:
            logger.warn("logger directory was not set. Ignore TFSummaryWriter.")
            return NoOpMonitor()

    def setup(self, trainer):
        super(TFSummaryWriter, self).setup(trainer)
        self._writer = tf.summary.FileWriter(logger.LOG_DIR, graph=tf.get_default_graph())

    def put_summary(self, summary):
        self._writer.add_summary(summary, self._trainer.global_step)

    def flush(self):
        self._writer.flush()

    def close(self):
        self._writer.close()


class JSONWriter(TrainingMonitor):
    """
    Write all scalar data to a json, grouped by their global step.
    """
    def __new__(cls):
        if logger.LOG_DIR:
            return super(JSONWriter, cls).__new__(cls)
        else:
            logger.warn("logger directory was not set. Ignore JSONWriter.")
            return NoOpMonitor()

    def setup(self, trainer):
        super(JSONWriter, self).setup(trainer)
        self._dir = logger.LOG_DIR
        self._fname = os.path.join(self._dir, 'stat.json')

        if os.path.isfile(self._fname):
            # TODO make a backup first?
            logger.info("Found existing JSON at {}, will append to it.".format(self._fname))
            with open(self._fname) as f:
                self._stats = json.load(f)
                assert isinstance(self._stats, list), type(self._stats)
        else:
            self._stats = []
        self._stat_now = {}

        self._last_gs = -1

    def put_scalar(self, name, val):
        gs = self._trainer.global_step
        if gs != self._last_gs:
            self._push()
            self._last_gs = gs
            self._stat_now['epoch_num'] = self._trainer.epoch_num
            self._stat_now['global_step'] = gs
        self._stat_now[name] = float(val)   # TODO will fail for non-numeric

    def _push(self):
        """ Note that this method is idempotent"""
        if len(self._stat_now):
            self._stats.append(self._stat_now)
            self._stat_now = {}
            self._write_stat()

    def _write_stat(self):
        tmp_filename = self._fname + '.tmp'
        try:
            with open(tmp_filename, 'w') as f:
                json.dump(self._stats, f)
            os.rename(tmp_filename, self._fname)
        except IOError:  # disk error sometimes..
            logger.exception("Exception in StatHolder.finalize()!")

    def flush(self):
        self._push()


# TODO print interval
class ScalarPrinter(TrainingMonitor):
    """
    Print all scalar data in terminal.
    """
    def __init__(self):
        self._whitelist = None
        self._blacklist = set([])

    def setup(self, _):
        self._dic = {}

    def put_scalar(self, name, val):
        self._dic[name] = float(val)

    def _print_stat(self):
        for k, v in sorted(self._dic.items(), key=operator.itemgetter(0)):
            if self._whitelist is None or k in self._whitelist:
                if k not in self._blacklist:
                    logger.info('{}: {:.5g}'.format(k, v))

    def flush(self):
        self._print_stat()
        self._dic = {}


class ScalarHistory(TrainingMonitor):
    """
    Only used by monitors internally.
    """
    def setup(self, _):
        self._dic = defaultdict(list)

    def put_scalar(self, name, val):
        self._dic[name].append(float(val))

    def get_latest(self, name):
        hist = self._dic[name]
        if len(hist) == 0:
            raise KeyError("Invalid key: {}".format(name))
        else:
            return hist[-1]

    def get_history(self, name):
        return self._dic[name]
