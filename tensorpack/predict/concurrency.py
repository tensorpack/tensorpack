#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: concurrency.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import multiprocessing
import six
from six.moves import queue, range
import tensorflow as tf

from ..utils import logger
from ..utils.develop import deprecated
from ..utils.concurrency import DIE, StoppableThread, ShareSessionThread
from ..tfutils.modelutils import describe_model
from .base import OnlinePredictor, OfflinePredictor, AsyncPredictorBase

__all__ = ['MultiProcessPredictWorker', 'MultiProcessQueuePredictWorker',
           'MultiThreadAsyncPredictor']


class MultiProcessPredictWorker(multiprocessing.Process):
    """ Base class for predict worker that runs offline in multiprocess"""

    def __init__(self, idx, config):
        """
        Args:
            idx (int): index of the worker. the 0th worker will print log.
            config (PredictConfig): the config to use.
        """
        super(MultiProcessPredictWorker, self).__init__()
        self.name = "MultiProcessPredictWorker-{}".format(idx)
        self.idx = idx
        self.config = config

    def _init_runtime(self):
        """ Call _init_runtime under different CUDA_VISIBLE_DEVICES, you'll
            have workers that run on multiGPUs
        """
        if self.idx != 0:
            from tensorpack.models.common import disable_layer_logging
            disable_layer_logging()
        self.predictor = OfflinePredictor(self.config)
        if self.idx == 0:
            with self.predictor.graph.as_default():
                describe_model()


class MultiProcessQueuePredictWorker(MultiProcessPredictWorker):
    """
    An offline predictor worker that takes input and produces output by queue.
    Each process will exit when they see :class:`DIE`.
    """

    def __init__(self, idx, inqueue, outqueue, config):
        """
        Args:
            idx, config: same as in :class:`MultiProcessPredictWorker`.
            inqueue (multiprocessing.Queue): input queue to get data point. elements are (task_id, dp)
            outqueue (multiprocessing.Queue): output queue to put result. elements are (task_id, output)
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
                self.outqueue.put((tid, self.predictor(dp)))


class PredictorWorkerThread(StoppableThread, ShareSessionThread):
    def __init__(self, queue, pred_func, id, batch_size=5):
        super(PredictorWorkerThread, self).__init__()
        self.name = "PredictorWorkerThread-{}".format(id)
        self.queue = queue
        self.func = pred_func
        self.daemon = True
        self.batch_size = batch_size
        self.id = id

    def run(self):
        with self.default_sess():
            while not self.stopped():
                batched, futures = self.fetch_batch()
                try:
                    outputs = self.func(batched)
                except tf.errors.CancelledError:
                    for f in futures:
                        f.cancel()
                    logger.warn("In PredictorWorkerThread id={}, call was cancelled.".format(self.id))
                    return
                # print "Worker {} batched {} Queue {}".format(
                #         self.id, len(futures), self.queue.qsize())
                #  debug, for speed testing
                # if not hasattr(self, 'xxx'):
                #     self.xxx = outputs = self.func(batched)
                # else:
                #     outputs = [[self.xxx[0][0]] * len(batched[0]), [self.xxx[1][0]] * len(batched[0])]

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
        while len(futures) < self.batch_size:
            try:
                inp, f = self.queue.get_nowait()
                for k in range(nr_input_var):
                    batched[k].append(inp[k])
                futures.append(f)
            except queue.Empty:
                break   # do not wait
        return batched, futures


class MultiThreadAsyncPredictor(AsyncPredictorBase):
    """
    An multithread online async predictor which runs a list of OnlinePredictor.
    It would do an extra batching internally.
    """

    def __init__(self, predictors, batch_size=5):
        """
        Args:
            predictors (list): a list of OnlinePredictor avaiable to use.
            batch_size (int): the maximum of an internal batch.
        """
        assert len(predictors)
        self._need_default_sess = False
        for k in predictors:
            assert isinstance(k, OnlinePredictor), type(k)
            if k.sess is None:
                self._need_default_sess = True
            # TODO support predictors.return_input here
            assert not k.return_input
        self.input_queue = queue.Queue(maxsize=len(predictors) * 100)
        self.threads = [
            PredictorWorkerThread(
                self.input_queue, f, id, batch_size=batch_size)
            for id, f in enumerate(predictors)]

        if six.PY2:
            # TODO XXX set logging here to avoid affecting TF logging
            import tornado.options as options
            options.parse_command_line(['--logging=debug'])

    def start(self):
        if self._need_default_sess:
            assert tf.get_default_session() is not None, \
                "Not session is bind to predictors, " \
                "MultiThreadAsyncPredictor.start() has to be called under a default session!"
        for t in self.threads:
            t.start()

    @deprecated("Use 'start()' instead!", "2017-03-11")
    def run(self):      # temporarily for back-compatibility
        self.start()

    def put_task(self, dp, callback=None):
        """
        Same as in :meth:`AsyncPredictorBase.put_task`.
        """
        f = Future()
        if callback is not None:
            f.add_done_callback(callback)
        self.input_queue.put((dp, f))
        return f


try:
    if six.PY2:
        from tornado.concurrent import Future
    else:
        from concurrent.futures import Future
except ImportError:
    from ..utils.develop import create_dummy_class
    MultiThreadAsyncPredictor = create_dummy_class('MultiThreadAsyncPredictor', 'tornado.concurrent')  # noqa
