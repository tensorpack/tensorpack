#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: base.py

import tensorflow as tf
import weakref
import time
from six.moves import range
import six

from ..callbacks import (
    Callback, Callbacks, Monitors, TrainingMonitor)
from ..utils import logger
from ..utils.argtools import call_only_once
from ..tfutils import get_global_step_value
from ..tfutils.tower import TowerFuncWrapper
from ..tfutils.model_utils import describe_trainable_vars
from ..tfutils.sessinit import JustCurrentSession
from ..tfutils.sesscreate import ReuseSessionCreator, NewSessionCreator
from ..callbacks.steps import MaintainStepCounter

from .config import TrainConfig, DEFAULT_MONITORS, DEFAULT_CALLBACKS

__all__ = ['StopTraining', 'Trainer']


class StopTraining(BaseException):
    """
    An exception thrown to stop training.
    """
    pass


class TrainLoop(object):
    """
    Manage the double for loop.
    """

    def __init__(self):
        self._epoch_num = 0
        self._global_step = 0
        self._local_step = -1

    def config(self, steps_per_epoch, starting_epoch, max_epoch):
        """
        Configure the loop given the settings.
        """
        self.starting_epoch = starting_epoch
        self.max_epoch = max_epoch
        self.steps_per_epoch = steps_per_epoch

        self._epoch_num = starting_epoch - 1

    def update_global_step(self):
        """
        Update the Python-side global_step from TF.
        This must be called under initialized default session.
        """
        self._global_step = get_global_step_value()

    @property
    def epoch_num(self):
        """
        The number of the currently ongoing epoch.

        An epoch is defined to cover the moment before calling `before_epoch` until after calling `trigger_epoch`.
        i.e., in the `trigger_epoch` of epoch 3, `self.epoch_num` is 3.
        If you need use `self.epoch_num` in your callback, you'll need to know this.
        """
        return self._epoch_num

    @property
    def global_step(self):
        """
        The tensorflow global_step, i.e. how many times ``hooked_sess.run`` has been called.

        Note:
            1. global_step is incremented **after** each ``hooked_sess.run`` returns from TF runtime.
            2. If you make zero or more than one calls to ``hooked_sess.run`` in one
               :meth:`run_step`, local_step and global_step may increment at different speed.
        """
        return self._global_step

    @property
    def local_step(self):
        """
        The number of steps that have finished in the current epoch.
        """
        return self._local_step


class Trainer(object):
    """ Base class for a trainer.
    """

    is_chief = True
    """
    Whether this process is the chief worker in distributed training.
    Certain callbacks will only be run by chief worker.
    """

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
        Register callbacks to the trainer.
        It can only be called before :meth:`Trainer.train()`.

        Args:
            cb (Callback or [Callback]): a callback or a list of callbacks
        """
        if isinstance(cb, (list, tuple)):
            for x in cb:
                self._register_callback(x)
            return
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

        1. Append :meth:`DEFAULT_CALLBACKS()` to callbacks.
        2. Append :meth:`DEFAULT_MONITORS()` to monitors.
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
                import tensorpack.trainv1 as old_train_mod    # noqa
                old_trainer = getattr(old_train_mod, name)
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
