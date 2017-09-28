# -*- coding: utf-8 -*-
# File: base.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import time
import weakref
from six.moves import range

import tensorflow as tf

from .config import TrainConfig
from ..utils import logger
from ..utils.naming import GLOBAL_STEP_INCR_OP_NAME
from ..callbacks import Callback, Callbacks
from ..callbacks.monitor import Monitors, TrainingMonitor
from ..tfutils import get_global_step_value, get_global_step_var
from ..tfutils.model_utils import describe_trainable_vars
from ..tfutils.sesscreate import ReuseSessionCreator
from ..tfutils.sessinit import JustCurrentSession

from ..graph_builder.predictor_factory import PredictorFactory

__all__ = ['Trainer', 'StopTraining']


class StopTraining(BaseException):
    """
    An exception thrown to stop training.
    """
    pass


class MaintainStepCounter(Callback):
    """
    It maintains the global step in the graph, making sure it's increased by one.
    This callback is always enabled by the trainer, you don't need to worry about it.
    """
    def _setup_graph(self):
        # ensure it exists
        gs_var = get_global_step_var()
        with tf.name_scope(None):
            with self.graph.colocate_with(gs_var):
                self.gs_incr_op = tf.assign_add(
                    gs_var, 1,
                    name=GLOBAL_STEP_INCR_OP_NAME).op
        self._fetches = tf.train.SessionRunArgs(self.gs_incr_op)

    def _before_train(self):
        if self.global_step != 0:
            logger.info("Start training with global_step={}".format(self.global_step))

    def _before_run(self, _):
        # always increase global_step when hooked_sess.run is called
        return self._fetches

    def _after_run(self, _, __):
        # Keep python-side global_step in agreement with TF-side
        self.trainer._global_step += 1


class Trainer(object):
    """ Base class for a trainer.

    Attributes:
        config (TrainConfig): the config used in this trainer.
        model (ModelDesc): alias for ``config.model``.
        sess (tf.Session): the current session in use.
        hooked_sess (tf.train.MonitoredSession): the session with hooks.
        monitors (Monitors): the monitors. Other callbacks can use it for logging.
        local_step (int): the number of (tensorpack) steps that have finished in the current epoch.
    """
    # step attr only available after before_train?

    is_chief = True

    def __init__(self, config):
        """
        Args:
            config (TrainConfig): the train config.
        """
        assert isinstance(config, TrainConfig), type(config)
        self.config = config
        self.model = config.model

        self.local_step = -1

        self._callbacks = []
        self.monitors = []
        self._epoch_num = None
        self._global_step = 0

        self._setup()   # subclass will setup the graph and InputSource

    @property
    def epoch_num(self):
        """
        The number of the currently ongoing epoch.

        An epoch is defined to cover the moment before calling `before_epoch` until after calling `trigger_epoch`.
        i.e., in the `trigger_epoch` of epoch 3, `self.epoch_num` is 3.
        If you need use `self.epoch_num` in your callback, you'll need to know this.
        """
        if self._epoch_num is not None:
            # has started training
            return self._epoch_num
        else:
            return self.config.starting_epoch - 1

    def register_callback(self, cb):
        """
        Register a callback to the trainer.
        It can only be called before :meth:`Trainer.train` gets called.
        """
        assert isinstance(cb, Callback), cb
        assert not isinstance(self._callbacks, Callbacks), \
            "Cannot register more callbacks after trainer was setup!"
        if not self.is_chief and cb.chief_only:
            logger.warn("Callback {} is chief-only, skipped.".format(str(cb)))
        else:
            self._callbacks.append(cb)

    def register_monitor(self, mon):
        """
        Register a monitor to the trainer.
        It can only be called before :meth:`Trainer.train` gets called.
        """
        assert isinstance(mon, TrainingMonitor), mon
        assert not isinstance(self.monitors, Monitors), \
            "Cannot register more monitors after trainer was setup!"
        if not self.is_chief and mon.chief_only:
            logger.warn("Monitor {} is chief-only, skipped.".format(str(mon)))
        else:
            self.monitors.append(mon)
            self.register_callback(mon)

    def train(self):
        """ Start training """
        self.setup()
        self.main_loop()

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

    def setup(self):
        """
        Setup the trainer and be ready for the main loop.
        """
        self.register_callback(MaintainStepCounter())
        for cb in self.config.callbacks:
            self.register_callback(cb)
        for m in self.config.monitors:
            self.register_monitor(m)
        self.monitors = Monitors(self.monitors)
        self.register_callback(self.monitors)

        describe_trainable_vars()

        # some final operations that might modify the graph
        logger.info("Setup callbacks graph ...")
        self._callbacks = Callbacks(self._callbacks)
        self._callbacks.setup_graph(weakref.proxy(self))

        logger.info("Creating the session ...")
        self._create_session()

        if self.is_chief:
            logger.info("Initializing the session ...")
            self.config.session_init.init(self.sess)
        else:
            assert isinstance(self.config.session_init, JustCurrentSession), \
                "session_init is only valid for chief worker session!"

        self.sess.graph.finalize()
        logger.info("Graph Finalized.")

    def _create_session(self):
        """
        Setup self.sess (the raw tf.Session)
        and self.hooked_sess (the session with hooks and coordinator)
        """
        hooks = self._callbacks.get_hooks()
        self.sess = self.config.session_creator.create_session()
        self.hooked_sess = tf.train.MonitoredSession(
            session_creator=ReuseSessionCreator(self.sess), hooks=hooks)

    def _setup(self):
        """
        Build the entire graph for training.
        Responsible for setup InputSource as well (including registering InputSource callbacks)

        Since this method will get called in constructor only,
        you can simply leave it empty and build your graph outside the trainer.
        """
        pass

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

    def main_loop(self):
        """
        Run the main training loop.
        """
        with self.sess.as_default():
            self._global_step = get_global_step_value()
            try:
                self._callbacks.before_train()
                # refresh global step (might have changed by callbacks) TODO ugly
                self._global_step = get_global_step_value()
                for self._epoch_num in range(
                        self.config.starting_epoch, self.config.max_epoch + 1):
                    logger.info("Start Epoch {} ...".format(self._epoch_num))
                    start_time = time.time()
                    self._callbacks.before_epoch()
                    for self.local_step in range(self.config.steps_per_epoch):
                        if self.hooked_sess.should_stop():
                            return
                        self.run_step()  # implemented by subclass
                        self._callbacks.trigger_step()
                    self._callbacks.after_epoch()
                    logger.info("Epoch {} (global_step {}) finished, time:{:.2f} sec.".format(
                        self._epoch_num, self.global_step, time.time() - start_time))

                    # trigger epoch outside the timing region.
                    self._callbacks.trigger_epoch()
                logger.info("Training has finished!")
            except (StopTraining, tf.errors.OutOfRangeError):
                logger.info("Training was stopped.")
            except KeyboardInterrupt:
                logger.info("Detected Ctrl-C and exiting main loop.")
            except:
                raise
            finally:
                self._callbacks.after_train()
                self.hooked_sess.close()

    # Predictor related methods. They actually should not be part of a trainer:
    @property
    def vs_name_for_predictor(self):
        # The variable scope name a predictor should be built in.
        # Expected to be changed. Don't use it.
        # TODO graphbuilder knows it
        return ""

    def get_predictor(self, input_names, output_names, tower=0):
        """
        Returns a callable predictor built under ``is_training=False`` tower context.
        Note that this method is only valid when this trainer has a ``ModelDesc``.

        Args:
            input_names (list), output_names(list): list of names
            tower (int): build the predictor on device '/gpu:{tower}' or use -1 for '/cpu:0'.

        Returns:
            an :class:`OnlinePredictor`.
        """
        # TODO move the logic to factory?
        return self.predictor_factory.get_predictor(input_names, output_names, tower)

    @property
    def predictor_factory(self):
        assert self.model is not None, \
            "Predictor can only be built one Trainer has ModelDesc!"
        if not hasattr(self, '_predictor_factory'):
            self._predictor_factory = PredictorFactory(
                self.model, self.vs_name_for_predictor)
        return self._predictor_factory
