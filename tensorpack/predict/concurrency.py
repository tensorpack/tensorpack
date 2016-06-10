#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: concurrency.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import multiprocessing, threading
import tensorflow as tf
import time
import six
from six.moves import queue, range, zip


from ..utils.concurrency import DIE
from ..tfutils.modelutils import describe_model
from ..utils import logger
from ..utils.timer import *
from ..tfutils import *

from .common import *

try:
    if six.PY2:
        from tornado.concurrent import Future
    else:
        from concurrent.futures import Future
except ImportError:
    logger.warn("Cannot import Future in either tornado.concurrent or py3 standard lib. MultiThreadAsyncPredictor won't be available.")
    __all__ = ['MultiProcessPredictWorker', 'MultiProcessQueuePredictWorker']
else:
    __all__ = ['MultiProcessPredictWorker', 'MultiProcessQueuePredictWorker',
                'MultiThreadAsyncPredictor']

class MultiProcessPredictWorker(multiprocessing.Process):
    """ Base class for predict worker that runs offline in multiprocess"""
    def __init__(self, idx, gpuid, config):
        """
        :param idx: index of the worker. the 0th worker will print log.
        :param gpuid: absolute id of the GPU to be used. set to -1 to use CPU.
        :param config: a `PredictConfig`
        """
        super(MultiProcessPredictWorker, self).__init__()
        self.idx = idx
        self.gpuid = gpuid
        self.config = config

    def _init_runtime(self):
        if self.gpuid >= 0:
            logger.info("Worker {} uses GPU {}".format(self.idx, self.gpuid))
            os.environ['CUDA_VISIBLE_DEVICES'] = str(self.gpuid)
        else:
            logger.info("Worker {} uses CPU".format(self.idx))
            os.environ['CUDA_VISIBLE_DEVICES'] = ''
        G = tf.Graph()     # build a graph for each process, because they don't need to share anything
        with G.as_default():
            if self.idx != 0:
                from tensorpack.models._common import disable_layer_logging
                disable_layer_logging()
            self.func = get_predict_func(self.config)
            if self.idx == 0:
                describe_model()

class MultiProcessQueuePredictWorker(MultiProcessPredictWorker):
    """ An offline predictor worker that takes input and produces output by queue"""
    def __init__(self, idx, gpuid, inqueue, outqueue, config):
        """
        :param inqueue: input queue to get data point. elements are (task_id, dp)
        :param outqueue: output queue put result. elements are (task_id, output)
        """
        super(MultiProcessQueuePredictWorker, self).__init__(idx, gpuid, config)
        self.inqueue = inqueue
        self.outqueue = outqueue

    def run(self):
        self._init_runtime()
        while True:
            tid, dp = self.inqueue.get()
            if tid == DIE:
                self.outqueue.put((DIE, None))
                return
            else:
                self.outqueue.put((tid, self.func(dp)))

class PredictorWorkerThread(threading.Thread):
    def __init__(self, queue, pred_func, id, nr_input_var, batch_size=5):
        super(PredictorWorkerThread, self).__init__()
        self.queue = queue
        self.func = pred_func
        self.daemon = True
        self.batch_size = batch_size
        self.nr_input_var = nr_input_var
        self.id = id

    def run(self):
        def fetch():
            batched, futures = [[] for _ in range(self.nr_input_var)], []
            inp, f = self.queue.get()
            for k in range(self.nr_input_var):
                batched[k].append(inp[k])
            futures.append(f)
            # fill a batch
            cnt = 1
            while cnt < self.batch_size:
                try:
                    inp, f = self.queue.get_nowait()
                    for k in range(self.nr_input_var):
                        batched[k].append(inp[k])
                    futures.append(f)
                    cnt += 1
                except queue.Empty:
                    break
            return batched, futures
        #self.xxx = None
        while True:
            batched, futures = fetch()
            #print "batched size: ", len(batched), "queuesize: ", self.queue.qsize()
            outputs = self.func(batched)
            # debug, for speed testing
            #if self.xxx is None:
                #outputs = self.func([batched])
                #self.xxx = outputs
            #else:
                #outputs = [None, None]
                #outputs[0] = [self.xxx[0][0]] * len(batched)
                #outputs[1] = [self.xxx[1][0]] * len(batched)

            for idx, f in enumerate(futures):
                f.set_result([k[idx] for k in outputs])

class MultiThreadAsyncPredictor(object):
    """
    An online predictor (use the current active session) that works with
    QueueInputTrainer. Use async interface, support multi-thread and multi-GPU.
    """
    def __init__(self, trainer, input_names, output_names, nr_thread, batch_size=5):
        """
        :param trainer: a `QueueInputTrainer` instance.
        """
        self.input_queue = queue.Queue(maxsize=nr_thread*10)
        self.threads = [
            PredictorWorkerThread(
                self.input_queue, f, id,
                len(input_names), batch_size=batch_size)
            for id, f in enumerate(
                trainer.get_predict_funcs(
                    input_names, output_names, nr_thread))]
        # TODO XXX set logging here to avoid affecting TF logging
        import tornado.options as options
        options.parse_command_line(['--logging=debug'])

    def run(self):
        for t in self.threads:
            t.start()

    def put_task(self, inputs, callback=None):
        """
        :param inputs: a data point (list of component) matching input_names (not batched)
        :param callback: a callback to get called with the list of outputs
        :returns: a Future of output."""
        f = Future()
        if callback is not None:
            f.add_done_callback(callback)
        self.input_queue.put((inputs, f))
        return f
