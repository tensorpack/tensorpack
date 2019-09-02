# -*- coding: utf-8 -*-
# File: config.py

import os
import tensorflow as tf

from ..callbacks import (
    JSONWriter, MergeAllSummaries, MovingAverageSummary, ProgressBar, RunUpdateOps, ScalarPrinter, TFEventWriter)
from ..dataflow.base import DataFlow
from ..input_source import InputSource
from ..tfutils.sesscreate import NewSessionCreator
from ..tfutils.sessinit import SaverRestore, SessionInit
from ..utils import logger

from .model_desc import ModelDescBase

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
    A collection of options to be used for single-cost trainers.

    Note that you do not have to use :class:`TrainConfig`.
    You can use the API of :class:`Trainer` directly, to have more fine-grained control of the training.
    """

    def __init__(self,
                 dataflow=None, data=None,
                 model=None,
                 callbacks=None, extra_callbacks=None, monitors=None,
                 session_creator=None, session_config=None, session_init=None,
                 starting_epoch=1, steps_per_epoch=None, max_epoch=99999,
                 **kwargs):
        """
        Args:
            dataflow (DataFlow):
            data (InputSource):
            model (ModelDesc):

            callbacks (list[Callback]): a list of :class:`Callback` to use during training.
            extra_callbacks (list[Callback]): This argument
                is only used to provide the defaults in addition to ``callbacks``.
                The list of callbacks that will be used in the end is simply ``callbacks + extra_callbacks``.

                It is usually left as None, and the default value for this argument is :func:`DEFAULT_CALLBACKS()`.
                You can override it when you don't like any of the default callbacks.
                For example, if you'd like to let the progress bar print tensors, you can use

                .. code-block:: none

                    extra_callbacks=[ProgressBar(names=['name']),
                                     MovingAverageSummary(),
                                     MergeAllSummaries(),
                                     RunUpdateOps()]

            monitors (list[MonitorBase]): Defaults to :func:`DEFAULT_MONITORS()`.

            session_creator (tf.train.SessionCreator): Defaults to :class:`sesscreate.NewSessionCreator()`
                with the config returned by :func:`tfutils.get_default_sess_config()`.
            session_config (tf.ConfigProto): when session_creator is None, use this to create the session.
            session_init (SessionInit): how to initialize variables of a session. Defaults to do nothing.

            starting_epoch (int): The index of the first epoch.
            steps_per_epoch (int): the number of steps (defined by :meth:`Trainer.run_step`) to run in each epoch.
                Defaults to the input data size. You may want to divide it by the #GPUs in multi-GPU training.

                Number of steps per epoch only affects the schedule of callbacks.
                It does not affect the sequence of input data seen by the model.
            max_epoch (int): maximum number of epoch to run training.
        """

        # TODO type checker decorator
        def assert_type(v, tp, name):
            assert isinstance(v, tp), \
                "{} has to be type '{}', but an object of type '{}' found.".format(
                    name, tp.__name__, v.__class__.__name__)

        # process data & model
        assert data is None or dataflow is None, "dataflow and data cannot be both presented in TrainConfig!"
        if dataflow is not None:
            assert_type(dataflow, DataFlow, 'dataflow')
        if data is not None:
            assert_type(data, InputSource, 'data')
        self.dataflow = dataflow
        self.data = data

        if model is not None:
            assert_type(model, ModelDescBase, 'model')
        self.model = model

        if callbacks is not None:
            assert_type(callbacks, list, 'callbacks')
        self.callbacks = callbacks
        if extra_callbacks is not None:
            assert_type(extra_callbacks, list, 'extra_callbacks')
        self.extra_callbacks = extra_callbacks
        if monitors is not None:
            assert_type(monitors, list, 'monitors')
        self.monitors = monitors
        if session_init is not None:
            assert_type(session_init, SessionInit, 'session_init')
        self.session_init = session_init

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
                    steps_per_epoch = len(dataflow)
                elif data is not None:
                    steps_per_epoch = data.size()
                else:
                    raise NotImplementedError()
            except NotImplementedError:
                logger.error("You must set `TrainConfig(steps_per_epoch)` if the size of your input is not available.")
                raise
        else:
            steps_per_epoch = int(steps_per_epoch)
        self.steps_per_epoch = steps_per_epoch

        self.starting_epoch = int(starting_epoch)
        self.max_epoch = int(max_epoch)


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

    Note that the functionality requires the logging directory to obtain
    necessary information from a previous run.
    If you have unconventional setup of logging directory, this class will not
    work for you, for example:

        1. If you save the checkpoint to a different directory rather than the
           logging directory.

        2. If in distributed training the directory is not
           available to every worker, or the directories are different for different workers.
    """
    def __init__(self, always_resume=True, **kwargs):
        """
        Args:
            always_resume (bool): If False, user-provided arguments
                `session_init` and `starting_epoch` will take priority.
                Otherwise, resume will take priority.
            kwargs: same as in :class:`TrainConfig`.

        Note:
            The main goal of this class is to let a training job resume
            without changing any line of code or command line arguments.
            So it's useful to let resume take priority over user-provided arguments sometimes.

            For example: if your training starts from a pre-trained model,
            you would want it to use user-provided model loader at the
            beginning, but a "resume" model loader when the job was
            interrupted and restarted.
        """
        found_sessinit = False
        if always_resume or 'session_init' not in kwargs:
            sessinit = self._get_sessinit_resume()
            if sessinit is not None:
                found_sessinit = True
                path = sessinit.path
                if 'session_init' in kwargs:
                    logger.info("Found checkpoint at {}. "
                                "session_init arguments will be overwritten.".format(path))
                else:
                    logger.info("Will load checkpoint at {}.".format(path))
                kwargs['session_init'] = sessinit

        found_last_epoch = False
        if always_resume or 'starting_epoch' not in kwargs:
            last_epoch = self._get_last_epoch()
            if last_epoch is not None:
                found_last_epoch = True
                now_epoch = last_epoch + 1
                logger.info("Found history statistics from JSON. "
                            "Setting starting_epoch to {}.".format(now_epoch))
                kwargs['starting_epoch'] = now_epoch
        assert found_sessinit == found_last_epoch, \
            "Found SessionInit={}, Found Last Epoch={}".format(found_sessinit, found_last_epoch)

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
