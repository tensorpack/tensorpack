# -*- coding: utf-8 -*-
# File: base.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import tensorflow as tf
from abc import ABCMeta, abstractmethod
from six.moves import range
import tqdm
import re

from .config import TrainConfig
from ..utils import *
from ..callbacks import StatHolder
from ..tfutils import *
from ..tfutils.modelutils import describe_model

__all__ = ['Trainer']

class Trainer(object):
    """
    Base class for a trainer.
    """
    __metaclass__ = ABCMeta

    def __init__(self, config):
        """
        :param config: a `TrainConfig` instance
        """
        assert isinstance(config, TrainConfig), type(config)
        self.config = config
        self.model = config.model

    @abstractmethod
    def train(self):
        """ Start training"""
        pass

    @abstractmethod
    def run_step(self):
        """ run an iteration"""
        pass

    def trigger_epoch(self):
        self._trigger_epoch()
        self.config.callbacks.trigger_epoch()
        self.summary_writer.flush()
        self.stat_holder.finalize()

    @abstractmethod
    def _trigger_epoch(self):
        """ This is called right after all steps in an epoch are finished"""
        pass

    def _init_summary(self):
        if not hasattr(logger, 'LOG_DIR'):
            raise RuntimeError("Please use logger.set_logger_dir at the beginning of your script.")
        self.summary_writer = tf.train.SummaryWriter(
            logger.LOG_DIR, graph=self.sess.graph)
        self.summary_op = tf.merge_all_summaries()
        # create an empty StatHolder
        self.stat_holder = StatHolder(logger.LOG_DIR)

    def _process_summary(self, summary_str):
        summary = tf.Summary.FromString(summary_str)
        for val in summary.value:
            if val.WhichOneof('value') == 'simple_value':
                val.tag = re.sub('tower[0-9]+/', '', val.tag)   # TODO move to subclasses
                self.stat_holder.add_stat(val.tag, val.simple_value)
        self.summary_writer.add_summary(summary, self.global_step)

    def main_loop(self):
        # some final operations that might modify the graph
        logger.info("Preparing for training...")
        self._init_summary()
        get_global_step_var()   # ensure there is such var, before finalizing the graph
        callbacks = self.config.callbacks
        callbacks.before_train(self)
        self.config.session_init.init(self.sess)
        tf.get_default_graph().finalize()
        self._start_all_threads()

        with self.sess.as_default():
            try:
                self.global_step = get_global_step()
                logger.info("Start training with global_step={}".format(self.global_step))

                for epoch in range(self.config.starting_epoch, self.config.max_epoch+1):
                    with timed_operation(
                        'Epoch {}, global_step={}'.format(
                            epoch, self.global_step + self.config.step_per_epoch)):
                        for step in tqdm.trange(
                                self.config.step_per_epoch,
                                leave=True, mininterval=0.5,
                                dynamic_ncols=True, ascii=True):
                            if self.coord.should_stop():
                                return
                            self.run_step()
                            #callbacks.trigger_step()   # not useful?
                            self.global_step += 1
                        self.trigger_epoch()
            except (KeyboardInterrupt, Exception):
                raise
            finally:
                self.coord.request_stop()
                # Do I need to run queue.close?
                callbacks.after_train()
                self.summary_writer.close()
                self.sess.close()

    def init_session_and_coord(self):
        describe_model()
        self.sess = tf.Session(config=self.config.session_config)
        self.coord = tf.train.Coordinator()

    def _start_all_threads(self):
        """
        Run all threads before starting training
        """
        tf.train.start_queue_runners(
            sess=self.sess, coord=self.coord, daemon=True, start=True)

    def process_grads(self, grads):
        procs = self.config.model.get_gradient_processor()
        for proc in procs:
            grads = proc.process(grads)
        return grads
