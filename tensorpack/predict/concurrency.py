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

from .base import *

try:
    if six.PY2:
        from tornado.concurrent import Future
    else:
        from concurrent.futures import Future
except ImportError:
    logger.warn("Cannot import Future in tornado.concurrent. MultiThreadAsyncPredictor won't be available.")
    __all__ = ['MultiProcessPredictWorker', 'MultiProcessQueuePredictWorker']
else:
    __all__ = ['MultiProcessPredictWorker', 'MultiProcessQueuePredictWorker',
                'MultiThreadAsyncPredictor']

class MultiProcessPredictWorker(multiprocessing.Process):
    """ Base class for predict worker that runs offline in multiprocess"""
    def __init__(self, idx, config):
        """
        :param idx: index of the worker. the 0th worker will print log.
        :param config: a `PredictConfig`
        """
        super(MultiProcessPredictWorker, self).__init__()
        self.idx = idx
        self.config = config

    def _init_runtime(self):
        """ Call _init_runtime under different CUDA_VISIBLE_DEVICES, you'll
            have workers that run on multiGPUs
        """
        if self.idx != 0:
            from tensorpack.models._common import disable_layer_logging
            disable_layer_logging()
        self.func = OfflinePredictor(self.config)
        if self.idx == 0:
            describe_model()

class MultiProcessQueuePredictWorker(MultiProcessPredictWorker):
    """ An offline predictor worker that takes input and produces output by queue"""
    def __init__(self, idx, inqueue, outqueue, config):
        """
        :param inqueue: input queue to get data point. elements are (task_id, dp)
        :param outqueue: output queue put result. elements are (task_id, output)
        """
        super(MultiProcessQueuePredictWorker, self).__init__(idx, config)
        self.inqueue = inqueue
        self.outqueue = outqueue
        assert isinstance(self.inqueue, multiprocessing.queues.Queue)
        assert isinstance(self.outqueue, multiprocessing.queues.Queue)

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
    def __init__(self, queue, pred_func, id, batch_size=5):
        super(PredictorWorkerThread, self).__init__()
        self.queue = queue
        self.func = pred_func
        self.daemon = True
        self.batch_size = batch_size
        self.id = id

    def run(self):
        while True:
            batched, futures = self.fetch_batch()
            outputs = self.func(batched)
            #print "Worker {} batched {} Queue {}".format(
                    #self.id, len(futures), self.queue.qsize())
            # debug, for speed testing
            #if not hasattr(self, 'xxx'):
                #self.xxx = outputs = self.func(batched)
            #else:
                #outputs = [[self.xxx[0][0]] * len(batched[0]), [self.xxx[1][0]] * len(batched[0])]

            for idx, f in enumerate(futures):
                f.set_result([k[idx] for k in outputs])

    def fetch_batch(self):
        """ Fetch a batch of data without waiting"""
        inp, f = self.queue.get()
        nr_input_var = len(inp)
        batched, futures = [[] for _ in range(nr_input_var)], []
        for k in range(nr_input_var):
            batched[k].append(inp[k])
        futures.append(f)
        cnt = 1
        while cnt < self.batch_size:
            try:
                inp, f = self.queue.get_nowait()
                for k in range(nr_input_var):
                    batched[k].append(inp[k])
                futures.append(f)
            except queue.Empty:
                break
            cnt += 1
        return batched, futures

class MultiThreadAsyncPredictor(AsyncPredictorBase):
    """
    An multithread online async predictor which run a list of PredictorBase.
    It would do an extra batching internally.
    """
    def __init__(self, predictors, batch_size=5):
        """ :param predictors: a list of OnlinePredictor"""
        assert len(predictors)
        for k in predictors:
            #assert isinstance(k, OnlinePredictor), type(k)
            # TODO use predictors.return_input here
            assert k.return_input == False
        self.input_queue = queue.Queue(maxsize=len(predictors)*100)
        self.threads = [
            PredictorWorkerThread(
                self.input_queue, f, id, batch_size=batch_size)
            for id, f in enumerate(predictors)]

        if six.PY2:
            # TODO XXX set logging here to avoid affecting TF logging
            import tornado.options as options
            options.parse_command_line(['--logging=debug'])

    def start(self):
        for t in self.threads:
            t.start()

    def run(self):      # temporarily for back-compatibility
        self.start()

    def put_task(self, dp, callback=None):
        """
        dp must be non-batched, i.e. single instance
        """
        f = Future()
        if callback is not None:
            f.add_done_callback(callback)
        self.input_queue.put((dp, f))
        return f
