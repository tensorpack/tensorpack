# -*- coding: utf-8 -*-
# File: base.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

from abc import ABCMeta, abstractmethod
import re
import time
import weakref
import six
from six.moves import range

import tensorflow as tf
from .predict import PredictorFactory
from .config import TrainConfig
from ..utils import logger, deprecated
from ..callbacks import StatHolder
from ..tfutils import get_global_step_value
from ..tfutils.modelutils import describe_model
from ..tfutils.summary import create_scalar_summary

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

        stat_holder (StatHolder)
        summary_writer (tf.summary.FileWriter)

        epoch_num (int): the number of epochs that have finished.
        local_step (int): the number of steps that have finished in the current epoch.
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

    def train(self):
        """ Start training """
        self.setup()
        self.main_loop()

    @abstractmethod
    def run_step(self):
        """ Abstract method: run one iteration. Subclass should define what is "iteration".
        """

    def trigger_epoch(self):
        """
        Called after each epoch.
        """
        # trigger subclass
        self._trigger_epoch()
        # trigger callbacks
        self.config.callbacks.trigger_epoch()
        self.summary_writer.flush()

    def _trigger_epoch(self):
        pass

    def add_summary(self, summary):
        """
        Add summary to ``self.summary_writer``, and also
        add scalar summary to ``self.stat_holder``.

        Args:
            summary (tf.Summary or str): a summary object, or a str which will
                be interpreted as a serialized tf.Summary protobuf.
        """
        if isinstance(summary, six.binary_type):
            summary = tf.Summary.FromString(summary)
        assert isinstance(summary, tf.Summary), type(summary)
        for val in summary.value:
            if val.WhichOneof('value') == 'simple_value':
                val.tag = re.sub('tower[p0-9]+/', '', val.tag)   # TODO move to subclasses
                suffix = '-summary'  # issue#6150
                if val.tag.endswith(suffix):
                    val.tag = val.tag[:-len(suffix)]
                self.stat_holder.add_stat(
                    val.tag, val.simple_value,
                    self.global_step, self.epoch_num)
        self.summary_writer.add_summary(summary, get_global_step_value())

    def add_scalar_summary(self, name, val):
        """
        Add a scalar summary to both TF events file and StatHolder.
        """
        self.add_summary(create_scalar_summary(name, val))

    def setup(self):
        """
        Setup the trainer and be ready for the main loop.
        """
        if not hasattr(logger, 'LOG_DIR'):
            raise RuntimeError("logger directory wasn't set!")

        self._setup()   # subclass will setup the graph

        describe_model()
        # some final operations that might modify the graph
        logger.info("Setup summaries ...")
        self.summary_writer = tf.summary.FileWriter(logger.LOG_DIR, graph=tf.get_default_graph())
        # create an empty StatHolder
        self.stat_holder = StatHolder(logger.LOG_DIR)

        logger.info("Setup callbacks graph ...")
        self.config.callbacks.setup_graph(weakref.proxy(self))
        self.config.session_init._setup_graph()

        def after_init(_, __):
            logger.info("Graph variables initialized.")
        scaffold = tf.train.Scaffold(
            init_op=tf.global_variables_initializer(),
            init_fn=after_init)
        logger.info("Finalize the graph, create the session ...")
        self.monitored_sess = tf.train.MonitoredSession(
            session_creator=tf.train.ChiefSessionCreator(
                scaffold=scaffold, config=self.config.session_config),
            hooks=self.config.callbacks.get_hooks())
        self.hooked_sess = self.monitored_sess  # just create an alias

        self.sess = self.monitored_sess._tf_sess()  # expose the underlying session also
        self.config.session_init._run_init(self.sess)

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
        callbacks = self.config.callbacks
        with self.sess.as_default():
            self._starting_step = get_global_step_value()
            try:
                callbacks.before_train()
                for self.epoch_num in range(
                        self.config.starting_epoch, self.config.max_epoch + 1):
                    logger.info("Start Epoch {} ...".format(self.epoch_num))
                    start_time = time.time()
                    for self.local_step in range(self.config.steps_per_epoch):
                        if self.monitored_sess.should_stop():
                            return
                        self.run_step()  # implemented by subclass
                        callbacks.trigger_step()
                    logger.info("Epoch {} (global_step {}) finished, time:{:.2f} sec.".format(
                        self.epoch_num, self.global_step, time.time() - start_time))

                    self.trigger_epoch()  # trigger epoch outside the timing region.
            except StopTraining:
                logger.info("Training was stopped.")
            except KeyboardInterrupt:
                logger.info("Detected Ctrl-C and exiting main loop.")
            except:
                raise
            finally:
                callbacks.after_train()
                self.summary_writer.close()
                self.monitored_sess.close()

    def get_predict_func(self, input_names, output_names, tower=0):
        """
        Args:
            input_names (list), output_names(list): list of names
            tower (int): return the predictor on the kth tower, defined by ``config.predict_tower``.

        Returns:
            an :class:`OnlinePredictor`.
        """
        if not hasattr(self, '_predictor_factory'):
            self._predictor_factory = PredictorFactory(
                self.model, self.config.predict_tower)
        return self._predictor_factory.get_predictor(input_names, output_names, tower)

    def get_predict_funcs(self, input_names, output_names, n):
        """ Return n predictors. """
        nr_tower = len(self.config.predict_tower)
        if nr_tower < n:
            logger.warn(
                "Requested {} predictor but only have {} predict towers! "
                "Predictors will be assigned to GPUs in round-robin.".format(n, nr_tower))
        return [self.get_predict_func(input_names, output_names, k % nr_tower) for k in range(n)]

    @deprecated("Don't need to call it any more!", "2017-03-20")
    def _setup_predictor_factory(self):
        pass
