#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: base.py

import tensorflow as tf
import weakref
import time
from six.moves import range
import six

from ..callbacks import (
    Callback, Callbacks, Monitors, TrainingMonitor,
    MovingAverageSummary,
    ProgressBar, MergeAllSummaries,
    TFEventWriter, JSONWriter, ScalarPrinter, RunUpdateOps)
from ..utils import logger
from ..utils.argtools import call_only_once
from ..tfutils.tower import TowerFuncWrapper
from ..tfutils.model_utils import describe_trainable_vars
from ..tfutils.sessinit import JustCurrentSession
from ..tfutils.sesscreate import ReuseSessionCreator, NewSessionCreator
from ..callbacks.steps import MaintainStepCounter

import tensorpack.trainv1 as old_train    # noqa
from ..trainv1.base import StopTraining, TrainLoop
from ..trainv1.config import TrainConfig

__all__ = ['StopTraining', 'TrainConfig',
           'Trainer', 'DEFAULT_MONITORS', 'DEFAULT_CALLBACKS']


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


class Trainer(object):
    """ Base class for a trainer.
    """

    _API_VERSION = 2

    is_chief = True

    def __init__(self, config=None):
        """
        config is only for compatibility reasons in case you're
        using custom trainers with old-style API.
        You should never use config.
        """
        self._callbacks = []
        self.loop = TrainLoop()

        # Hacks!
        if config is not None:
            logger.warn("You're initializing new trainer with old trainer API!")
            logger.warn("This could happen if you wrote a custom trainer before.")
            logger.warn("It may work now through some hacks, but please switch to the new API!")
            logger.warn("See https://github.com/ppwwyyxx/tensorpack/issues/458 for more information.")
            self._config = config
            self.inputs_desc = config.model.get_inputs_desc()
            self.tower_func = TowerFuncWrapper(
                lambda *inputs: config.model.build_graph(*inputs),
                self.inputs_desc)
            self._main_tower_vs_name = ""

            def gp(input_names, output_names, tower=0):
                from .tower import TowerTrainer
                return TowerTrainer.get_predictor(self, input_names, output_names, device=tower)
            self.get_predictor = gp

            old_train = self.train

            def train():
                return old_train(
                    config.callbacks, config.monitors,
                    config.session_creator, config.session_init,
                    config.steps_per_epoch, config.starting_epoch, config.max_epoch)
            self.train = train

    def _register_callback(self, cb):
        """
        Register a callback to the trainer.
        It can only be called before :meth:`Trainer.train()`.
        """
        assert isinstance(cb, Callback), cb
        assert not isinstance(self._callbacks, Callbacks), \
            "Cannot register more callbacks after trainer was setup!"
        if not self.is_chief and cb.chief_only:
            logger.warn("Callback {} is chief-only, skipped.".format(str(cb)))
        else:
            self._callbacks.append(cb)

    register_callback = _register_callback

    def run_step(self):
        """
        Defines what to do in one iteration. The default is:
        ``self.hooked_sess.run(self.train_op)``.

        The behavior can be changed by either defining what is ``train_op``,
        or overriding this method.
        """
        if not hasattr(self, 'train_op'):
            raise NotImplementedError(
                "Please either set `Trainer.train_op` or provide an implementation "
                "of Trainer.run_step()!")
        self.hooked_sess.run(self.train_op)

    @call_only_once
    def setup_callbacks(self, callbacks, monitors):
        """
        Setup callbacks and monitors. Must be called after the main graph is built.

        Args:
            callbacks ([Callback]):
            monitors ([TrainingMonitor]):
        """
        describe_trainable_vars()   # TODO weird

        self.register_callback(MaintainStepCounter())
        for cb in callbacks:
            self.register_callback(cb)
        for cb in self._callbacks:
            assert not isinstance(cb, TrainingMonitor), "Monitor cannot be pre-registered for now!"
        for m in monitors:
            self.register_callback(m)
        self.monitors = Monitors(monitors)
        self.register_callback(self.monitors)   # monitors is also a callback

        # some final operations that might modify the graph
        logger.info("Setup callbacks graph ...")
        self._callbacks = Callbacks(self._callbacks)
        self._callbacks.setup_graph(weakref.proxy(self))

    @call_only_once
    def initialize(self, session_creator, session_init):
        """
        Initialize self.sess and self.hooked_sess.
        Must be called after callbacks are setup.

        Args:
            session_creator (tf.train.SessionCreator):
            session_init (sessinit.SessionInit):
        """
        session_init._setup_graph()

        logger.info("Creating the session ...")

        hooks = self._callbacks.get_hooks()
        self.sess = session_creator.create_session()
        self.hooked_sess = tf.train.MonitoredSession(
            session_creator=ReuseSessionCreator(self.sess), hooks=hooks)

        if self.is_chief:
            logger.info("Initializing the session ...")
            session_init._run_init(self.sess)
        else:
            if not isinstance(session_init, JustCurrentSession):
                logger.warn("This is not a chief worker, 'session_init' was ignored!")

        self.sess.graph.finalize()
        logger.info("Graph Finalized.")

    @call_only_once
    def main_loop(self, steps_per_epoch, starting_epoch, max_epoch):
        """
        Run the main training loop.

        Args:
            steps_per_epoch, starting_epoch, max_epoch (int):
        """
        with self.sess.as_default():
            self.loop.config(steps_per_epoch, starting_epoch, max_epoch)
            self.loop.update_global_step()
            try:
                self._callbacks.before_train()
                # refresh global step (might have changed by callbacks) TODO ugly
                # what if gs is changed later?
                self.loop.update_global_step()
                for self.loop._epoch_num in range(
                        self.loop.starting_epoch, self.loop.max_epoch + 1):
                    logger.info("Start Epoch {} ...".format(self.loop.epoch_num))
                    start_time = time.time()
                    self._callbacks.before_epoch()
                    for self.loop._local_step in range(self.loop.steps_per_epoch):
                        if self.hooked_sess.should_stop():
                            return
                        self.run_step()  # implemented by subclass
                        self._callbacks.trigger_step()
                    self._callbacks.after_epoch()
                    logger.info("Epoch {} (global_step {}) finished, time:{:.2f} sec.".format(
                        self.loop.epoch_num, self.loop.global_step, time.time() - start_time))

                    # trigger epoch outside the timing region.
                    self._callbacks.trigger_epoch()
                logger.info("Training has finished!")
            except (StopTraining, tf.errors.OutOfRangeError):
                logger.info("Training was stopped.")
            except KeyboardInterrupt:
                logger.info("Detected Ctrl-C and exiting main loop.")
                raise
            finally:
                self._callbacks.after_train()
                self.hooked_sess.close()

    def train(self,
              callbacks, monitors,
              session_creator, session_init,
              steps_per_epoch, starting_epoch=1, max_epoch=9999999):
        """
        Implemented by:

        .. code-block:: python

            self.setup_callbacks(callbacks, monitors)
            self.initialize(session_creator, session_init)
            self.main_loop(steps_per_epoch, starting_epoch, max_epoch)

        You can call those methods by yourself to have better control on details if needed.
        """
        self.setup_callbacks(callbacks, monitors)
        self.initialize(session_creator, session_init)
        self.main_loop(steps_per_epoch, starting_epoch, max_epoch)

    def train_with_defaults(
            self, callbacks=None, monitors=None,
            session_creator=None, session_init=None,
            steps_per_epoch=None, starting_epoch=1, max_epoch=9999999):
        """
        Same as :meth:`train()`, but will:

        1. Append `DEFAULT_CALLBACKS()` to callbacks.
        2. Append `DEFAULT_MONITORS()` to monitors.
        3. Provide default values for every option except `steps_per_epoch`.
        """
        callbacks = (callbacks or []) + DEFAULT_CALLBACKS()
        monitors = (monitors or []) + DEFAULT_MONITORS()

        assert steps_per_epoch is not None
        session_creator = session_creator or NewSessionCreator()
        session_init = session_init or JustCurrentSession()

        self.train(callbacks, monitors,
                   session_creator, session_init,
                   steps_per_epoch, starting_epoch, max_epoch)

    # create the old trainer when called with TrainConfig
    def __new__(cls, *args, **kwargs):
        if (len(args) > 0 and isinstance(args[0], TrainConfig)) \
                or 'config' in kwargs:
            name = cls.__name__
            try:
                old_trainer = getattr(old_train, name)
            except AttributeError:
                # custom trainer. has to live with it
                return super(Trainer, cls).__new__(cls)
            else:
                logger.warn("You're calling new trainers with old trainer API!")
                logger.warn("Now it returns the old trainer for you, please switch to use new trainers soon!")
                logger.warn("See https://github.com/ppwwyyxx/tensorpack/issues/458 for more information.")
                return old_trainer(*args, **kwargs)
        else:
            return super(Trainer, cls).__new__(cls)


def _get_property(name):
    """
    Delegate property to self.loop
    """
    ret = property(
        lambda self: getattr(self.loop, name))
    if six.PY3:     # __doc__ is readonly in Py2
        try:
            ret.__doc__ = getattr(TrainLoop, name).__doc__
        except AttributeError:
            pass
    return ret


for name in ['global_step', 'local_step', 'steps_per_epoch',
             'epoch_num', 'starting_epoch', 'max_epoch']:
    setattr(Trainer, name, _get_property(name))
