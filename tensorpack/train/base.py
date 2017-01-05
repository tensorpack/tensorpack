# -*- coding: utf-8 -*-
# File: base.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

from abc import ABCMeta, abstractmethod
import re
import weakref
import six
from six.moves import range
import tqdm

import tensorflow as tf
from .config import TrainConfig
from ..utils import logger, get_tqdm_kwargs
from ..utils.timer import timed_operation
from ..callbacks import StatHolder
from ..tfutils import get_global_step, get_global_step_var
from ..tfutils.modelutils import describe_model
from ..tfutils.summary import create_summary

__all__ = ['Trainer', 'StopTraining']


class StopTraining(BaseException):
    pass


@six.add_metaclass(ABCMeta)
class Trainer(object):
    """ Base class for a trainer."""

    """a `StatHolder` instance"""
    stat_holder = None

    """`tf.SummaryWriter`"""
    summary_writer = None
    """a tf.Tensor which returns summary string"""
    summary_op = None
    """ TrainConfig """
    config = None
    """ a ModelDesc"""
    model = None
    """ the current session"""
    sess = None
    """ the `tf.train.Coordinator` """
    coord = None

    def __init__(self, config):
        """
        :param config: a `TrainConfig` instance
        """
        assert isinstance(config, TrainConfig), type(config)
        self.config = config
        self.model = config.model
        self.sess = tf.Session(config=self.config.session_config)
        self.coord = tf.train.Coordinator()

    def train(self):
        """ Start training"""
        self.setup()
        self.main_loop()

    @abstractmethod
    def run_step(self):
        """ run an iteration"""
        pass

    def get_predict_func(self, input_names, output_names):
        """ return a online predictor"""
        raise NotImplementedError()

    def get_predict_funcs(self, input_names, output_names, n):
        """ return n predictor functions.
            Can be overwritten by subclasses to exploit more
            parallelism among funcs.
        """
        return [self.get_predict_func(input_names, output_names) for k in range(n)]

    def trigger_epoch(self):
        # trigger subclass
        self._trigger_epoch()
        # trigger callbacks
        self.config.callbacks.trigger_epoch()
        self.summary_writer.flush()

    @abstractmethod
    def _trigger_epoch(self):
        """ This is called right after all steps in an epoch are finished"""
        pass

    def _process_summary(self, summary_str):
        summary = tf.Summary.FromString(summary_str)
        for val in summary.value:
            if val.WhichOneof('value') == 'simple_value':
                val.tag = re.sub('tower[p0-9]+/', '', val.tag)   # TODO move to subclasses
                suffix = '-summary'  # issue#6150
                if val.tag.endswith(suffix):
                    val.tag = val.tag[:-len(suffix)]
                self.stat_holder.add_stat(val.tag, val.simple_value)
        self.summary_writer.add_summary(summary, get_global_step())

    def write_scalar_summary(self, name, val):
        self.summary_writer.add_summary(
            create_summary(name, val), get_global_step())
        self.stat_holder.add_stat(name, val)

    def setup(self):
        self._setup()
        describe_model()
        get_global_step_var()
        # some final operations that might modify the graph
        logger.info("Setup callbacks ...")
        self.config.callbacks.setup_graph(weakref.proxy(self))

        if not hasattr(logger, 'LOG_DIR'):
            raise RuntimeError("logger directory wasn't set!")
        self.summary_writer = tf.summary.FileWriter(logger.LOG_DIR, graph=self.sess.graph)
        self.summary_op = tf.summary.merge_all()
        # create an empty StatHolder
        self.stat_holder = StatHolder(logger.LOG_DIR)

        logger.info("Initializing graph variables ...")
        # TODO newsession + sessinit?
        initop = tf.global_variables_initializer()
        self.sess.run(initop)
        self.config.session_init.init(self.sess)

        tf.get_default_graph().finalize()
        tf.train.start_queue_runners(
            sess=self.sess, coord=self.coord, daemon=True, start=True)

    @abstractmethod
    def _setup(self):
        """ setup Trainer-specific stuff for training"""

    def main_loop(self):
        callbacks = self.config.callbacks
        with self.sess.as_default():
            try:
                callbacks.before_train()
                logger.info("Start training with global_step={}".format(get_global_step()))
                for epoch_num in range(
                        self.config.starting_epoch, self.config.max_epoch + 1):
                    with timed_operation(
                        'Epoch {} (global_step {})'.format(
                            epoch_num, get_global_step() + self.config.step_per_epoch)):
                        for step in tqdm.trange(
                                self.config.step_per_epoch,
                                **get_tqdm_kwargs(leave=True)):
                            if self.coord.should_stop():
                                return
                            self.run_step()  # implemented by subclass
                            callbacks.trigger_step()   # not useful?
                    # trigger epoch outside the timing region.
                    self.trigger_epoch()
            except StopTraining:
                logger.info("Training was stopped.")
            except KeyboardInterrupt:
                logger.info("Detected Ctrl+C and shutdown training.")
            except:
                raise
            finally:
                callbacks.after_train()
                self.coord.request_stop()
                self.summary_writer.close()
                self.sess.close()
