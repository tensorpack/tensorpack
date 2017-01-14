#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: dataset.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

from six.moves import range, zip
from abc import ABCMeta, abstractmethod
import multiprocessing
import os
import six

from ..dataflow import DataFlow
from ..dataflow.dftools import dataflow_to_process_queue
from ..utils.concurrency import ensure_proc_terminate, OrderedResultGatherProc, DIE
from ..utils import logger, get_tqdm
from ..utils.gpu import change_gpu

from .concurrency import MultiProcessQueuePredictWorker
from .config import PredictConfig
from .base import OfflinePredictor

__all__ = ['DatasetPredictorBase', 'SimpleDatasetPredictor',
           'MultiProcessDatasetPredictor']


@six.add_metaclass(ABCMeta)
class DatasetPredictorBase(object):
    """ Base class for dataset predictors.
        These are predictors which run over a :class:`DataFlow`.
    """

    def __init__(self, config, dataset):
        """
        Args:
            config (PredictConfig): the config of predictor.
            dataset (DataFlow): the DataFlow to run on.
        """
        assert isinstance(dataset, DataFlow)
        assert isinstance(config, PredictConfig)
        self.config = config
        self.dataset = dataset

    @abstractmethod
    def get_result(self):
        """
        Yields:
            output for each datapoint in the DataFlow.
        """
        pass

    def get_all_result(self):
        """
        Returns:
            list: all outputs for all datapoints in the DataFlow.
        """
        return list(self.get_result())


class SimpleDatasetPredictor(DatasetPredictorBase):
    """
    Simply create one predictor and run it on the DataFlow.
    """
    def __init__(self, config, dataset):
        super(SimpleDatasetPredictor, self).__init__(config, dataset)
        self.predictor = OfflinePredictor(config)

    def get_result(self):
        self.dataset.reset_state()
        try:
            sz = self.dataset.size()
        except NotImplementedError:
            sz = 0
        with get_tqdm(total=sz, disable=(sz == 0)) as pbar:
            for dp in self.dataset.get_data():
                res = self.predictor(dp)
                yield res
                pbar.update()


class MultiProcessDatasetPredictor(DatasetPredictorBase):
    """
    Run prediction in multiprocesses, on either CPU or GPU.
    Each process fetch datapoints as tasks and run predictions independently.
    """
    # TODO allow unordered

    def __init__(self, config, dataset, nr_proc, use_gpu=True, ordered=True):
        """
        Args:
            config: same as in :class:`DatasetPredictorBase`.
            dataset: same as in :class:`DatasetPredictorBase`.
            nr_proc (int): number of processes to use
            use_gpu (bool): use GPU or CPU.
                If GPU, then ``nr_proc`` cannot be more than what's in
                CUDA_VISIBLE_DEVICES.
            ordered (bool): produce outputs in the original order of the
                datapoints. This will be a bit slower. Otherwise, :meth:`get_result` will produce
                outputs in any order.
        """
        if config.return_input:
            logger.warn("Using the option `return_input` in MultiProcessDatasetPredictor might be slow")
        assert nr_proc > 1, nr_proc
        super(MultiProcessDatasetPredictor, self).__init__(config, dataset)

        self.nr_proc = nr_proc
        self.ordered = ordered

        self.inqueue, self.inqueue_proc = dataflow_to_process_queue(
            self.dataset, nr_proc * 2, self.nr_proc)    # put (idx, dp) to inqueue

        if use_gpu:
            try:
                gpus = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
                assert len(gpus) >= self.nr_proc, \
                    "nr_proc={} while only {} gpus available".format(
                    self.nr_proc, len(gpus))
            except KeyError:
                # TODO number of GPUs not checked
                gpus = list(range(self.nr_proc))
        else:
            gpus = ['-1'] * self.nr_proc
        # worker produces (idx, result) to outqueue
        self.outqueue = multiprocessing.Queue()
        self.workers = [MultiProcessQueuePredictWorker(
            i, self.inqueue, self.outqueue, self.config)
            for i in range(self.nr_proc)]

        # start inqueue and workers
        self.inqueue_proc.start()
        for p, gpuid in zip(self.workers, gpus):
            if gpuid == '-1':
                logger.info("Worker {} uses CPU".format(p.idx))
            else:
                logger.info("Worker {} uses GPU {}".format(p.idx, gpuid))
            with change_gpu(gpuid):
                p.start()

        if ordered:
            self.result_queue = OrderedResultGatherProc(
                self.outqueue, nr_producer=self.nr_proc)
            self.result_queue.start()
            ensure_proc_terminate(self.result_queue)
        else:
            self.result_queue = self.outqueue
        ensure_proc_terminate(self.workers + [self.inqueue_proc])

    def get_result(self):
        try:
            sz = self.dataset.size()
        except NotImplementedError:
            sz = 0
        with get_tqdm(total=sz, disable=(sz == 0)) as pbar:
            die_cnt = 0
            while True:
                res = self.result_queue.get()
                pbar.update()
                if res[0] != DIE:
                    yield res[1]
                else:
                    die_cnt += 1
                    if die_cnt == self.nr_proc:
                        break
        self.inqueue_proc.join()
        self.inqueue_proc.terminate()
        if self.ordered:    # if ordered, than result_queue is a Process
            self.result_queue.join()
            self.result_queue.terminate()
        for p in self.workers:
            p.join()
            p.terminate()
