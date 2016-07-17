# -*- coding: UTF-8 -*-
# File: trainer.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import tensorflow as tf
import threading
import time
from six.moves import zip

from .base import Trainer

from ..dataflow.common import RepeatedData
from ..tfutils.summary import summary_moving_average
from ..tfutils.modelutils import describe_model

from ..utils import *
from ..tfutils import *
from ..tfutils.summary import add_moving_summary
from ..predict import OnlinePredictor

__all__ = ['SimpleTrainer', 'QueueInputTrainer']

class PredictorFactory(object):
    """ Make predictors for a trainer"""
    PREFIX = 'towerp'

    def __init__(self, trainer, towers):
        self.trainer = trainer
        self.towers = towers
        self.tower_built = False

    def get_predictor(self, input_names, output_names, tower):
        if not self.tower_built:
            self._build_predict_tower()
        tower = self.towers[tower % len(self.towers)]
        raw_input_vars = get_vars_by_names(input_names)
        output_names = ['{}{}/'.format(self.PREFIX, tower) + n for n in output_names]
        output_vars = get_vars_by_names(output_names)
        return OnlinePredictor(self.trainer.sess, raw_input_vars, output_vars)

    def _build_predict_tower(self):
        # build_predict_tower might get called anywhere, but 'towerp' should be the outermost name scope
        with tf.name_scope(None), \
                freeze_collection(SUMMARY_BACKUP_KEYS):
            inputs = self.trainer.model.get_input_vars()
            tf.get_variable_scope().reuse_variables()
            for k in self.towers:
                logger.info("Building graph for predictor tower {}...".format(k))
                with tf.device('/gpu:{}'.format(k) if k >= 0 else '/cpu:0'), \
                        tf.name_scope('{}{}'.format(self.PREFIX, k)):
                    self.trainer.model.build_graph(inputs, False)
        self.tower_built = True

class SimpleTrainer(Trainer):
    def run_step(self):
        data = next(self.data_producer)
        feed = dict(zip(self.input_vars, data))
        self.sess.run([self.train_op], feed_dict=feed)    # faster since train_op return None

    def train(self):
        model = self.model
        self.input_vars = model.get_input_vars()
        model.build_graph(self.input_vars, True)
        cost_var = model.get_cost()
        add_moving_summary(cost_var)

        grads = self.config.optimizer.compute_gradients(cost_var)
        grads = self.process_grads(grads)

        avg_maintain_op = summary_moving_average()
        self.train_op = tf.group(
            self.config.optimizer.apply_gradients(grads, get_global_step_var()),
            avg_maintain_op)

        self.init_session_and_coord()
        describe_model()
        # create an infinte data producer
        self.config.dataset.reset_state()
        self.data_producer = RepeatedData(self.config.dataset, -1).get_data()
        self.main_loop()

    def _trigger_epoch(self):
        if self.summary_op is not None:
            data = next(self.data_producer)
            feed = dict(zip(self.input_vars, data))
            summary_str = self.summary_op.eval(feed_dict=feed)
            self._process_summary(summary_str)

    def get_predict_func(self, input_names, output_names):
        if not hasattr(self, 'predictor_factory'):
            self.predictor_factory = PredictorFactory(self, [0])
        return self.predictor_factory.get_predictor(input_names, output_names, 0)

class EnqueueThread(threading.Thread):
    def __init__(self, trainer):
        super(EnqueueThread, self).__init__()
        self.sess = trainer.sess
        self.coord = trainer.coord
        self.dataflow = RepeatedData(trainer.config.dataset, -1)

        self.input_vars = trainer.input_vars
        self.queue = trainer.input_queue
        self.op = self.queue.enqueue(self.input_vars)
        self.close_op = self.queue.close(cancel_pending_enqueues=True)

        self.size_op = self.queue.size()
        self.daemon = True

    def run(self):
        self.dataflow.reset_state()
        with self.sess.as_default():
            try:
                while True:
                    for dp in self.dataflow.get_data():
                        if self.coord.should_stop():
                            return
                        feed = dict(zip(self.input_vars, dp))
                        #print 'TFQ:', self.sess.run([self.op, self.size_op], feed_dict=feed)[1]
                        self.op.run(feed_dict=feed)
            except tf.errors.CancelledError as e:
                pass
            except Exception:
                logger.exception("Exception in EnqueueThread:")
            finally:
                try:
                    self.sess.run(self.close_op)
                except RuntimeError:    # session already closed
                    pass
                self.coord.request_stop()
                logger.info("Enqueue Thread Exited.")

class QueueInputTrainer(Trainer):
    """ Single GPU Trainer, takes input from a queue"""

    def __init__(self, config, input_queue=None, predict_tower=None):
        """
        :param config: a `TrainConfig` instance
        :param input_queue: a `tf.QueueBase` instance to be used to buffer datapoints.
            Defaults to a FIFO queue of size 100.
        :param predict_tower: list of gpu idx to run prediction. default to be [0].
            Use -1 for cpu.
        """
        super(QueueInputTrainer, self).__init__(config)
        self.input_vars = self.model.get_input_vars()
        # use a smaller queue size for now, to avoid https://github.com/tensorflow/tensorflow/issues/2942
        if input_queue is None:
            self.input_queue = tf.FIFOQueue(
                    50, [x.dtype for x in self.input_vars], name='input_queue')
        else:
            self.input_queue = input_queue

        if predict_tower is None:
            # by default, use the first training gpu for prediction
            predict_tower = [0]
        self.predictor_factory = PredictorFactory(self, predict_tower)

        self.dequed_inputs = None

    def _get_model_inputs(self):
        """ Dequeue a datapoint from input_queue and return"""
        ret = self.input_queue.dequeue(name='input_deque')
        if isinstance(ret, tf.Tensor): # only one input
            ret = [ret]
        assert len(ret) == len(self.input_vars)
        for qv, v in zip(ret, self.input_vars):
            qv.set_shape(v.get_shape())
        return ret

    def _single_tower_grad(self):
        """ Get grad and cost for single-tower"""
        self.dequed_inputs = model_inputs = self._get_model_inputs()

        # test the overhead of queue
        #with tf.device('/gpu:0'):
            #self.dequed_inputs = [tf.Variable(tf.random_normal([128,224,224,3],
                #dtype=tf.float32), trainable=False),
                #tf.Variable(tf.ones([128], dtype=tf.int32), trainable=False)]
        self.model.build_graph(self.dequed_inputs, True)
        cost_var = self.model.get_cost()
        grads = self.config.optimizer.compute_gradients(
                cost_var, gate_gradients=0) # GATE_NONE
        add_moving_summary(cost_var)
        return grads

    def _build_enque_thread(self):
        """ create a thread that keeps filling the queue """
        self.input_th = EnqueueThread(self)
        self._extra_threads_procs.append(self.input_th)

    def train(self):
        assert self.config.nr_tower == 1, \
                "QueueInputTrainer doesn't support multigpu! Use Sync/AsyncMultiGPUTrainer instead."
        self.init_session_and_coord()
        self._build_enque_thread()

        grads = self._single_tower_grad()
        grads = self.process_grads(grads)
        describe_model()

        self.train_op = tf.group(
            self.config.optimizer.apply_gradients(grads, get_global_step_var()),
            summary_moving_average(), name='train_op')

        self.main_loop()

    def run_step(self):
        """ just run self.train_op"""
        self.sess.run(self.train_op)
        #run_metadata = tf.RunMetadata()
        #self.sess.run([self.train_op],
                #options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
                #run_metadata=run_metadata
                #)
        #from tensorflow.python.client import timeline
        #trace = timeline.Timeline(step_stats=run_metadata.step_stats)
        #trace_file = open('timeline.ctf.json', 'w')
        #trace_file.write(trace.generate_chrome_trace_format())
        #import sys; sys.exit()

        #self.sess.run([self.dequed_inputs[1]])

    def _trigger_epoch(self):
        # need to run summary_op every epoch
        # note that summary_op will take a data from the queue
        if self.summary_op is not None:
            summary_str = self.summary_op.eval()
            self._process_summary(summary_str)

    def get_predict_func(self, input_names, output_names, tower=0):
        """
        :param tower: return the kth predict_func
        :returns: an `OnlinePredictor`
        """
        return self.predictor_factory.get_predictor(input_names, output_names, tower)

