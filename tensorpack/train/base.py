# -*- coding: utf-8 -*-
# File: base.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

from abc import ABCMeta, abstractmethod
import time
import weakref
import six
from six.moves import range

import tensorflow as tf

from ..graph_builder.predictor_factory import PredictorFactory
from .config import TrainConfig
from ..utils import logger
from ..callbacks import Callback, Callbacks, MaintainStepCounter
from ..callbacks.monitor import Monitors, TrainingMonitor
from ..tfutils import get_global_step_value
from ..tfutils.model_utils import describe_trainable_vars
from ..tfutils.sesscreate import ReuseSessionCreator
from ..tfutils.sessinit import JustCurrentSession

__all__ = ['Trainer', 'StopTraining']


class StopTraining(BaseException):
    """
    An exception thrown to stop training.
    """
    pass


@six.add_metaclass(ABCMeta)
class Trainer(object):
    """ Base class for a trainer.

    Attributes:
        config (TrainConfig): the config used in this trainer.
        model (ModelDesc)
        sess (tf.Session): the current session in use.
        hooked_sess (tf.MonitoredSession): the session with hooks.
        monitors (Monitors): the monitors. Callbacks can use it for logging.

        epoch_num (int): the number of epochs that have finished.
        local_step (int): the number of steps that have finished in the current epoch.
        global_step (int): the number of steps that have finished or is currently running.
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

        self._setup()   # subclass will setup the graph and InputSource

    @property
    def epoch_num(self):
        if self._epoch_num is not None:
            # has started training
            return self._epoch_num
        else:
            return self.config.starting_epoch - 1

    def register_callback(self, cb):
        """
        Use this method before :meth:`Trainer._setup` finishes,
        to register a callback to the trainer.
        """
        assert isinstance(cb, Callback), cb
        assert not isinstance(self._callbacks, Callbacks), \
            "Cannot register more callbacks after trainer was setup!"
        if not self.is_chief and cb.chief_only:
            logger.warn("Callback {} is chief-only, skipped.".format(str(cb)))
        else:
            self._callbacks.append(cb)

    def register_monitor(self, mon):
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

    @abstractmethod
    def run_step(self):
        """ Abstract method: run one iteration. Subclass should define what is "iteration".
        """

    def _setup_input_source(self, input_source):
        """
        Setup InputSource on this trainer.
        """
        input_source.setup(self.model.get_inputs_desc())
        cbs = input_source.get_callbacks()
        self.config.callbacks.extend(cbs)

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
        try:
            return self._starting_step + \
                self.config.steps_per_epoch * (self.epoch_num - 1) + \
                self.local_step + 1  # +1: the ongoing step
        except AttributeError:
            return get_global_step_value()

    def main_loop(self):
        """
        Run the main training loop.
        """
        with self.sess.as_default():
            self._starting_step = get_global_step_value()
            try:
                self._callbacks.before_train()
                # refresh global step (might have changed by callbacks) TODO ugly
                self._starting_step = get_global_step_value()
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

    # Predictor related methods:
    @property
    def vs_name_for_predictor(self):
        """
        The variable scope name a predictor should be built in.
        """
        # TODO graphbuilder knows it
        return ""

    def get_predictor(self, input_names, output_names, tower=0):
        """
        Args:
            input_names (list), output_names(list): list of names
            tower (int): return the predictor on the kth tower, defined by ``config.predict_tower``.

        Returns:
            an :class:`OnlinePredictor`.
        """
        # TODO move the logic to factory?
        nr_tower = len(self.config.predict_tower)
        if nr_tower < tower:
            logger.warn(
                "Requested the {}th predictor but only have {} predict towers! "
                "Predictors will be assigned to GPUs in round-robin.".format(tower, nr_tower))
        tower = tower % nr_tower
        return self.predictor_factory.get_predictor(input_names, output_names, tower)

    @property
    def predictor_factory(self):
        if not hasattr(self, '_predictor_factory'):
            self._predictor_factory = PredictorFactory(
                self.model, self.config.predict_tower, self.vs_name_for_predictor)
        return self._predictor_factory

    def get_predictors(self, input_names, output_names, n):
        """ Return n predictors. """
        return [self.get_predictor(input_names, output_names, k) for k in range(n)]
