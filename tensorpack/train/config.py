#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: config.py

import os
import tensorflow as tf

from ..callbacks import (
    MovingAverageSummary,
    ProgressBar, MergeAllSummaries,
    TFEventWriter, JSONWriter, ScalarPrinter, RunUpdateOps)
from ..dataflow.base import DataFlow
from ..graph_builder.model_desc import ModelDescBase
from ..utils import logger
from ..tfutils.sessinit import JustCurrentSession, SessionInit, SaverRestore
from ..tfutils.sesscreate import NewSessionCreator
from ..input_source import InputSource

__all__ = ['TrainConfig', 'AutoResumeTrainConfig', 'DEFAULT_CALLBACKS', 'DEFAULT_MONITORS']


def DEFAULT_CALLBACKS():
    """
    Return the default callbacks,
    which will be used in :class:`TrainConfig` and :meth:`Trainer.train_with_defaults`.
    They are:

    1. MovingAverageSummary()
    2. ProgressBar()
    3. MergeAllSummaries()
    4. RunUpdateOps()
    """
    return [
        MovingAverageSummary(),
        ProgressBar(),
        MergeAllSummaries(),
        RunUpdateOps()]


def DEFAULT_MONITORS():
    """
    Return the default monitors,
    which will be used in :class:`TrainConfig` and :meth:`Trainer.train_with_defaults`.
    They are:

    1. TFEventWriter()
    2. JSONWriter()
    3. ScalarPrinter()
    """
    return [TFEventWriter(), JSONWriter(), ScalarPrinter()]


class TrainConfig(object):
    """
    A collection of options to be used for trainers.
    """

    def __init__(self,
                 dataflow=None, data=None, model=None,
                 callbacks=None, extra_callbacks=None, monitors=None,
                 session_creator=None, session_config=None, session_init=None,
                 starting_epoch=1, steps_per_epoch=None, max_epoch=99999,
                 nr_tower=1, tower=None,
                 **kwargs):
        """
        Args:
            dataflow (DataFlow):
            data (InputSource):
            model (ModelDescBase):

            callbacks (list): a list of :class:`Callback` to perform during training.
            extra_callbacks (list): the same as ``callbacks``. This argument
                is only used to provide the defaults in addition to ``callbacks``.
                The list of callbacks that will be used in the end is ``callbacks + extra_callbacks``.

                It is usually left as None and the default value for this
                option will be the return value of :meth:`train.DEFAULT_CALLBACKS()`.
                You can override it when you don't like any of the default callbacks.
            monitors (list): a list of :class:`TrainingMonitor`.
                Defaults to the return value of :meth:`train.DEFAULT_MONITORS()`.

            session_creator (tf.train.SessionCreator): Defaults to :class:`sesscreate.NewSessionCreator()`
                with the config returned by :func:`tfutils.get_default_sess_config()`.
            session_config (tf.ConfigProto): when session_creator is None, use this to create the session.
            session_init (SessionInit): how to initialize variables of a session. Defaults to do nothing.

            starting_epoch (int): The index of the first epoch.
            steps_per_epoch (int): the number of steps (defined by :meth:`Trainer.run_step`) to run in each epoch.
                Defaults to the input data size.
            max_epoch (int): maximum number of epoch to run training.
        """

        # TODO type checker decorator
        def assert_type(v, tp):
            assert isinstance(v, tp), v.__class__

        # process data & model
        assert data is None or dataflow is None, "dataflow and data cannot be both presented in TrainConfig!"
        if dataflow is not None:
            assert_type(dataflow, DataFlow)
        if data is not None:
            assert_type(data, InputSource)
        self.dataflow = dataflow
        self.data = data

        if model is not None:
            assert_type(model, ModelDescBase)
        self.model = model

        if callbacks is None:
            callbacks = []
        assert_type(callbacks, list)
        if extra_callbacks is not None:
            self._callbacks = callbacks + extra_callbacks
        else:
            self._callbacks = callbacks + DEFAULT_CALLBACKS()

        self.monitors = monitors if monitors is not None else DEFAULT_MONITORS()

        if session_init is None:
            session_init = JustCurrentSession()
        self.session_init = session_init
        assert_type(self.session_init, SessionInit)

        if session_creator is None:
            if session_config is not None:
                self.session_creator = NewSessionCreator(config=session_config)
            else:
                self.session_creator = NewSessionCreator(config=None)
        else:
            self.session_creator = session_creator
            assert session_config is None, "Cannot set both session_creator and session_config!"

        if steps_per_epoch is None:
            try:
                if dataflow is not None:
                    steps_per_epoch = dataflow.size()
                elif data is not None:
                    steps_per_epoch = data.size()
                else:
                    raise NotImplementedError()
            except NotImplementedError:
                logger.error("You must set `TrainConfig(steps_per_epoch)` if data.size() is not available.")
                raise
        else:
            steps_per_epoch = int(steps_per_epoch)
        self.steps_per_epoch = steps_per_epoch

        self.starting_epoch = int(starting_epoch)
        self.max_epoch = int(max_epoch)

        # Tower stuff are for Trainer v1 only:
        nr_tower = max(nr_tower, 1)
        self.nr_tower = nr_tower
        if tower is not None:
            assert self.nr_tower == 1, "Cannot set both nr_tower and tower in TrainConfig!"
            self.tower = tower

        assert len(kwargs) == 0, 'Unknown arguments: {}'.format(str(kwargs.keys()))

    @property
    def nr_tower(self):
        return len(self.tower)

    @nr_tower.setter
    def nr_tower(self, value):
        self.tower = list(range(value))

    @property
    def callbacks(self):        # disable setter
        return self._callbacks


class AutoResumeTrainConfig(TrainConfig):
    """
    Same as :class:`TrainConfig`, but does the following to automatically
    resume from training:

    1. If a checkpoint was found in :meth:`logger.get_logger_dir()`, set
       `session_init` option to load it.
    2. If a JSON history was found in :meth:`logger.get_logger_dir()`, try to
       load the epoch number from it and set the `starting_epoch` option to
       continue training.

    You can choose to let the above two option to either overwrite or
    not overwrite user-provided arguments, as explained below.
    """
    def __init__(self, always_resume=True, **kwargs):
        """
        Args:
            always_resume (bool): If False, user-provided arguments
                `session_init` and `starting_epoch` will take priority.
                Otherwise, resume will take priority.
            kwargs: same as in :class:`TrainConfig`.

        Notes:
            The main goal of this class is to let a training job to resume
            without changing any line of code or command line arguments.
            So it's useful to let resume take priority over user-provided arguments sometimes:

            If your training starts from a pretrained model,
            you would want it to use user-provided model loader at the
            beginning, but a "resume" model loader when the job was
            interrupted and restarted.
        """
        if always_resume or 'session_init' not in kwargs:
            sessinit = self._get_sessinit_resume()
            if sessinit is not None:
                path = sessinit.path
                if 'session_init' in kwargs:
                    logger.info("Found checkpoint at {}. "
                                "session_init arguments will be overwritten.".format(path))
                else:
                    logger.info("Will load checkpoint at {}.".format(path))
                kwargs['session_init'] = sessinit

        if always_resume or 'starting_epoch' not in kwargs:
            last_epoch = self._get_last_epoch()
            if last_epoch is not None:
                now_epoch = last_epoch + 1
                logger.info("Found history statistics from JSON. "
                            "Overwrite the starting epoch to epoch #{}.".format(now_epoch))
                kwargs['starting_epoch'] = now_epoch

        super(AutoResumeTrainConfig, self).__init__(**kwargs)

    def _get_sessinit_resume(self):
        logdir = logger.get_logger_dir()
        if not logdir:
            return None
        path = os.path.join(logdir, 'checkpoint')
        if not tf.gfile.Exists(path):
            return None
        return SaverRestore(path)

    def _get_last_epoch(self):
        return JSONWriter.load_existing_epoch_number()
