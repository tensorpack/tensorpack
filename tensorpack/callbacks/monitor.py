# -*- coding: utf-8 -*-
# File: monitor.py


import json
import numpy as np
import operator
import os
import re
import shutil
import time
from collections import defaultdict
from datetime import datetime
import six
import threading

from ..compat import tfv1 as tf
from ..libinfo import __git_version__
from ..tfutils.summary import create_image_summary, create_scalar_summary
from ..utils import fs, logger
from ..utils.develop import HIDE_DOC
from .base import Callback

__all__ = ['MonitorBase', 'Monitors',
           'TFEventWriter', 'JSONWriter',
           'ScalarPrinter', 'SendMonitorData',
           'CometMLMonitor']


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


class MonitorBase(Callback):
    """
    Base class for monitors which monitor a training progress, by processing different types of
    summary/statistics from trainer.

    .. document private functions
    .. automethod:: _setup_graph
    """

    _chief_only = False

    def setup_graph(self, trainer):
        # Set attributes following Callback.setup_graph
        self.trainer = trainer
        self.graph = tf.get_default_graph()
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


class NoOpMonitor(MonitorBase):
    def __init__(self, name=None):
        self._name = name

    def __str__(self):
        if self._name is None:
            return "NoOpMonitor"
        return "NoOpMonitor({})".format(self._name)


class Monitors(Callback):
    """
    Merge monitors together for trainer to use.

    In training, each trainer will create a :class:`Monitors` instance,
    and you can access it through ``trainer.monitors``.
    You should use ``trainer.monitors`` for logging and it will dispatch your
    logs to each sub-monitor.
    """

    _chief_only = False

    def __init__(self, monitors):
        self._scalar_history = ScalarHistory()
        self._monitors = monitors + [self._scalar_history]
        for m in self._monitors:
            assert isinstance(m, MonitorBase), m

    def _setup_graph(self):
        # scalar_history's other methods were not called.
        # but they are not useful for now
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

                # TODO This hack is still needed, seem to disappear only when
                # compiled from source.
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
                in range [0,255]. If channel is 3, assumed to be RGB.
        """
        assert isinstance(val, np.ndarray)
        arr = image_to_nhwc(val)
        self._dispatch(lambda m: m.process_image(name, arr))
        s = create_image_summary(name, arr)
        self._dispatch(lambda m: m.process_summary(s))

    def put_event(self, evt):
        """
        Put an :class:`tf.Event`.
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

        If you run multiprocess training, keep in mind that
        the data is perhaps only available on chief process.

        Returns:
            scalar
        """
        return self._scalar_history.get_latest(name)[1]

    def get_history(self, name):
        """
        Get a history of the scalar value of some data.

        If you run multiprocess training, keep in mind that
        the data is perhaps only available on chief process.

        Returns:
            a list of (global_step, value) pairs: history data for this scalar
        """
        return self._scalar_history.get_history(name)


class TFEventWriter(MonitorBase):
    """
    Write summaries to TensorFlow event file.
    """
    def __init__(self, logdir=None, max_queue=10, flush_secs=120, split_files=False):
        """
        Args:
            logdir: ``logger.get_logger_dir()`` by default.
            max_queue, flush_secs: Same as in :class:`tf.summary.FileWriter`.
            split_files: if True, split events to multiple files rather than
                append to a single file. Useful on certain filesystems where append is expensive.
        """
        if logdir is None:
            logdir = logger.get_logger_dir()
        assert tf.gfile.IsDirectory(logdir), logdir
        self._logdir = fs.normpath(logdir)
        self._max_queue = max_queue
        self._flush_secs = flush_secs
        self._split_files = split_files

    def __new__(cls, logdir=None, max_queue=10, flush_secs=120, **kwargs):
        if logdir is None:
            logdir = logger.get_logger_dir()

        if logdir is not None:
            return super(TFEventWriter, cls).__new__(cls)
        else:
            logger.warn("logger directory was not set. Ignore TFEventWriter.")
            return NoOpMonitor("TFEventWriter")

    def _setup_graph(self):
        self._writer = tf.summary.FileWriter(
            self._logdir, max_queue=self._max_queue, flush_secs=self._flush_secs)

    def _write_graph(self):
        self._writer.add_graph(self.graph)

    def _before_train(self):
        # Writing the graph is expensive (takes ~2min) when the graph is large.
        # Therefore use a separate thread. It will then run in the
        # background while TF is warming up in the first several iterations.
        self._write_graph_thread = threading.Thread(target=self._write_graph)
        self._write_graph_thread.daemon = True
        self._write_graph_thread.start()

    @HIDE_DOC
    def process_summary(self, summary):
        self._writer.add_summary(summary, self.global_step)

    @HIDE_DOC
    def process_event(self, evt):
        self._writer.add_event(evt)

    def _trigger(self):     # flush every epoch
        self._writer.flush()
        if self._split_files:
            self._writer.close()
            self._writer.reopen()  # open new file

    def _after_train(self):
        self._writer.close()


class JSONWriter(MonitorBase):
    """
    Write all scalar data to a json file under ``logger.get_logger_dir()``, grouped by their global step.
    If found an earlier json history file, will append to it.
    """

    FILENAME = 'stats.json'
    """
    The name of the json file. Do not change it.
    """

    def __new__(cls):
        if logger.get_logger_dir():
            return super(JSONWriter, cls).__new__(cls)
        else:
            logger.warn("logger directory was not set. Ignore JSONWriter.")
            return NoOpMonitor("JSONWriter")

    @staticmethod
    def load_existing_json():
        """
        Look for an existing json under :meth:`logger.get_logger_dir()` named "stats.json",
        and return the loaded list of statistics if found. Returns None otherwise.
        """
        dir = logger.get_logger_dir()
        fname = os.path.join(dir, JSONWriter.FILENAME)
        if tf.gfile.Exists(fname):
            with open(fname) as f:
                stats = json.load(f)
                assert isinstance(stats, list), type(stats)
                return stats
        return None

    @staticmethod
    def load_existing_epoch_number():
        """
        Try to load the latest epoch number from an existing json stats file (if any).
        Returns None if not found.
        """
        stats = JSONWriter.load_existing_json()
        try:
            return int(stats[-1]['epoch_num'])
        except Exception:
            return None

    # initialize the stats here, because before_train from other callbacks may use it
    def _setup_graph(self):
        self._stats = []
        self._stat_now = {}
        self._last_gs = -1

    def _before_train(self):
        stats = JSONWriter.load_existing_json()
        self._fname = os.path.join(logger.get_logger_dir(), JSONWriter.FILENAME)
        if stats is not None:
            try:
                epoch = stats[-1]['epoch_num'] + 1
            except Exception:
                epoch = None

            # check against the current training settings
            # therefore this logic needs to be in before_train stage
            starting_epoch = self.trainer.loop.starting_epoch
            if epoch is None or epoch == starting_epoch:
                logger.info("Found existing JSON inside {}, will append to it.".format(logger.get_logger_dir()))
                self._stats = stats
            else:
                logger.warn(
                    "History epoch={} from JSON is not the predecessor of the current starting_epoch={}".format(
                        epoch - 1, starting_epoch))
                logger.warn("If you want to resume old training, either use `AutoResumeTrainConfig` "
                            "or correctly set the new starting_epoch yourself to avoid inconsistency. ")

                backup_fname = JSONWriter.FILENAME + '.' + datetime.now().strftime('%m%d-%H%M%S')
                backup_fname = os.path.join(logger.get_logger_dir(), backup_fname)

                logger.warn("Now, we will train with starting_epoch={} and backup old json to {}".format(
                    self.trainer.loop.starting_epoch, backup_fname))
                shutil.move(self._fname, backup_fname)

        # in case we have something to log here.
        self._trigger()

    def _trigger_step(self):
        # will do this in trigger_epoch
        if self.local_step != self.trainer.steps_per_epoch - 1:
            self._trigger()

    def _trigger_epoch(self):
        self._trigger()

    @HIDE_DOC
    def process_scalar(self, name, val):
        self._stat_now[name] = val

    def _trigger(self):
        """
        Add stats to json and dump to disk.
        Note that this method is idempotent.
        """
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


class ScalarPrinter(MonitorBase):
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
            rs = {re.compile(r) for r in rs}
            return rs

        self._whitelist = compile_regex(whitelist)
        if blacklist is None:
            blacklist = []
        self._blacklist = compile_regex(blacklist)

        self._enable_step = enable_step
        self._enable_epoch = enable_epoch
        self._dic = {}

    # in case we have something to log here.
    def _before_train(self):
        self._trigger()

    def _trigger_step(self):
        if self._enable_step:
            if self.local_step != self.trainer.steps_per_epoch - 1:
                # not the last step
                self._trigger()
            else:
                if not self._enable_epoch:
                    self._trigger()
                # otherwise, will print them together

    def _trigger_epoch(self):
        if self._enable_epoch:
            self._trigger()

    @HIDE_DOC
    def process_scalar(self, name, val):
        self._dic[name] = float(val)

    def _trigger(self):
        # Print stats here
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


class ScalarHistory(MonitorBase):
    """
    Only internally used by monitors.
    """

    def __init__(self):
        self._dic = defaultdict(list)

    @HIDE_DOC
    def process_scalar(self, name, val):
        self._dic[name].append((self.global_step, float(val)))

    def get_latest(self, name):
        hist = self._dic[name]
        if len(hist) == 0:
            raise KeyError("No available data for the key: {}".format(name))
        else:
            return hist[-1]

    def get_history(self, name):
        return self._dic[name]


class SendMonitorData(MonitorBase):
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

    @HIDE_DOC
    def process_scalar(self, name, val):
        if name in self.names:
            self.dic[name] = val

    def _trigger_step(self):
        self._trigger()

    def _trigger(self):
        try:
            v = {k: self.dic[k] for k in self.names}
        except KeyError:
            return
        cmd = self.command.format(**v)
        ret = os.system(cmd)
        if ret != 0:
            logger.error("Command '{}' failed with ret={}!".format(cmd, ret))
        self.dic = {}


class CometMLMonitor(MonitorBase):
    """
    Send scalar data and the graph to https://www.comet.ml.

    Note:
        1. comet_ml requires you to `import comet_ml` before importing tensorflow or tensorpack.
        2. The "automatic output logging" feature of comet_ml will make the training progress bar appear to freeze.
           Therefore the feature is disabled by default.
    """
    def __init__(self, experiment=None, tags=None, **kwargs):
        """
        Args:
            experiment (comet_ml.Experiment): if provided, invalidate all other arguments
            tags (list[str]): experiment tags
            kwargs: arguments used to initialize :class:`comet_ml.Experiment`,
                such as project name, API key, etc.
                Refer to its documentation for details.
        """
        if experiment is not None:
            self._exp = experiment
            assert tags is None and len(kwargs) == 0
        else:
            from comet_ml import Experiment
            kwargs.setdefault('log_code', True)  # though it's not functioning, git patch logging requires it
            kwargs.setdefault('auto_output_logging', None)
            self._exp = Experiment(**kwargs)
            if tags is not None:
                self._exp.add_tags(tags)

        self._exp.set_code("Code logging is impossible ...")
        self._exp.log_dependency('tensorpack', __git_version__)

    @property
    def experiment(self):
        """
        The :class:`comet_ml.Experiment` instance.
        """
        return self._exp

    def _before_train(self):
        self._exp.set_model_graph(tf.get_default_graph())

    @HIDE_DOC
    def process_scalar(self, name, val):
        self._exp.log_metric(name, val, step=self.global_step)

    @HIDE_DOC
    def process_image(self, name, val):
        self._exp.set_step(self.global_step)
        for idx, v in enumerate(val):
            log_name = "{}_step{}{}".format(
                name,
                self.global_step,
                "_" + str(idx) if len(val) > 1 else "")

            self._exp.log_image(v, image_format="jpeg", name=log_name, image_minmax=(0, 255))

    def _after_train(self):
        self._exp.end()

    def _after_epoch(self):
        self._exp.log_epoch_end(self.epoch_num)
