# -*- coding: utf-8 -*-
# File: base.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

from abc import ABCMeta, abstractmethod
import time
import weakref
import six
from six.moves import range

import tensorflow as tf
from tensorflow.python.training.monitored_session \
    import _HookedSession as HookedSession

from .predict import PredictorFactory
from .config import TrainConfig
from ..utils import logger
from ..utils.develop import deprecated
from ..callbacks import Callback, Callbacks, MaintainStepCounter
from ..callbacks.monitor import Monitors, TrainingMonitor
from ..tfutils import get_global_step_value
from ..tfutils.model_utils import describe_model
from ..tfutils.sesscreate import ReuseSessionCreator

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
        monitors (Monitors): the monitors. Callbacks can use it for logging.

        epoch_num (int): the number of epochs that have finished.
        local_step (int): the number of steps that have finished in the current epoch.
        global_step (int): the number of steps that have finished.
    """

    def __init__(self, config):
        """
        Args:
            config (TrainConfig): the train config.
        """
        assert isinstance(config, TrainConfig), type(config)
        self.config = config
        self.model = config.model

        self.epoch_num = self.config.starting_epoch - 1
        self.local_step = -1

        self._callbacks = []
        self.register_callback(MaintainStepCounter())
        for cb in config.callbacks:
            self.register_callback(cb)

        self.monitors = []
        for m in config.monitors:
            self.register_monitor(m)

    def register_callback(self, cb):
        """
        Use this method before :meth:`Trainer._setup` finishes,
        to register a callback to the trainer.

        The hooks of the registered callback will be bind to the
        `self.hooked_sess` session.
        """
        assert isinstance(cb, Callback), cb
        assert not isinstance(self._callbacks, Callbacks), \
            "Cannot register more callbacks after trainer was setup!"
        self._callbacks.append(cb)

    def register_monitor(self, mon):
        assert isinstance(mon, TrainingMonitor), mon
        assert not isinstance(self.monitors, Monitors), \
            "Cannot register more monitors after trainer was setup!"
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

    def _trigger_epoch(self):
        pass

    def setup(self):
        """
        Setup the trainer and be ready for the main loop.
        """
        self._setup()   # subclass will setup the graph

        self.monitors = Monitors(self.monitors)
        self.register_callback(self.monitors)

        describe_model()

        # some final operations that might modify the graph
        logger.info("Setup callbacks graph ...")
        self._callbacks = Callbacks(self._callbacks)
        self._callbacks.setup_graph(weakref.proxy(self))

        # create session
        logger.info("Finalize the graph, create the session ...")
        self.sess = self.config.session_creator.create_session()
        self._monitored_sess = tf.train.MonitoredSession(
            session_creator=ReuseSessionCreator(self.sess), hooks=None)

        # init session
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        logger.info("Graph variables initialized.")
        self.config.session_init.init(self.sess)
        # self.sess.graph.finalize()

        hooks = self._callbacks.get_hooks()
        self.hooked_sess = HookedSession(self.sess, hooks)

    @abstractmethod
    def _setup(self):
        """ setup Trainer-specific stuff for training"""

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
                for self.epoch_num in range(
                        self.config.starting_epoch, self.config.max_epoch + 1):
                    logger.info("Start Epoch {} ...".format(self.epoch_num))
                    start_time = time.time()
                    for self.local_step in range(self.config.steps_per_epoch):
                        if self._monitored_sess.should_stop():
                            return
                        self.run_step()  # implemented by subclass
                        self._callbacks.trigger_step()
                    logger.info("Epoch {} (global_step {}) finished, time:{:.2f} sec.".format(
                        self.epoch_num, self.global_step, time.time() - start_time))

                    # trigger epoch outside the timing region.
                    self._trigger_epoch()
                    self._callbacks.trigger_epoch()
            except (StopTraining, tf.errors.OutOfRangeError):
                logger.info("Training was stopped.")
            except KeyboardInterrupt:
                logger.info("Detected Ctrl-C and exiting main loop.")
            except:
                raise
            finally:
                self._callbacks.after_train()
                self._monitored_sess.close()

    # Predictor related methods:    TODO
    def get_predictor(self, input_names, output_names, tower=0):
        """
        Args:
            input_names (list), output_names(list): list of names
            tower (int): return the predictor on the kth tower, defined by ``config.predict_tower``.

        Returns:
            an :class:`OnlinePredictor`.
        """
        if not hasattr(self, '_predictor_factory'):
            self._predictor_factory = PredictorFactory(self)
        nr_tower = len(self.config.predict_tower)
        if nr_tower < tower:
            logger.warn(
                "Requested the {}th predictor but only have {} predict towers! "
                "Predictors will be assigned to GPUs in round-robin.".format(tower, nr_tower))
        tower = tower % nr_tower
        return self._predictor_factory.get_predictor(input_names, output_names, tower)

    def get_predictors(self, input_names, output_names, n):
        """ Return n predictors. """
        return [self.get_predictor(input_names, output_names, k) for k in range(n)]

    @deprecated("Use get_predictor instead!", "2017-05-20")
    def get_predict_func(self, input_names, output_names, tower=0):
        return self.get_predictor(input_names, output_names, tower)

    @deprecated("Use get_predictors instead!", "2017-05-20")
    def get_predict_funcs(self, input_names, output_names, n):
        return self.get_predictors(input_names, output_names, n)
