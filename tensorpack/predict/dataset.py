#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: dataset.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

from six.moves import range
from tqdm import tqdm
from abc import ABCMeta, abstractmethod
import multiprocessing
import os

from ..dataflow import DataFlow, BatchData
from ..dataflow.dftools import dataflow_to_process_queue
from ..utils.concurrency import ensure_proc_terminate, OrderedResultGatherProc, DIE

from .concurrency import MultiProcessQueuePredictWorker
from .common import *

__all__ = ['DatasetPredictorBase', 'SimpleDatasetPredictor',
        'MultiProcessDatasetPredictor']

class DatasetPredictorBase(object):
    __metaclass__ = ABCMeta

    def __init__(self, config, dataset):
        """
        :param config: a `PredictConfig` instance.
        :param dataset: a `DataFlow` instance.
        """
        assert isinstance(dataset, DataFlow)
        assert isinstance(config, PredictConfig)
        self.config = config
        self.dataset = dataset

    @abstractmethod
    def get_result(self):
        """ Generate (inpupt, output) pair of output, for each input in dataset"""
        pass

    def get_all_result(self):
        """
        Run over the dataset and return a list of all predictions.
        """
        return list(self.get_result())

class SimpleDatasetPredictor(DatasetPredictorBase):
    """
    Run the predict_config on a given `DataFlow`.
    """
    def __init__(self, config, dataset):
        super(SimpleDatasetPredictor, self).__init__(config, dataset)
        self.func = get_predict_func(config)

    def get_result(self):
        """ A generator to produce prediction for each data"""
        with tqdm(total=self.dataset.size()) as pbar:
            for dp in self.dataset.get_data():
                res = self.func(dp)
                if self.config.return_input:
                    yield (dp, res)
                else:
                    yield res
                pbar.update()

class MultiProcessDatasetPredictor(DatasetPredictorBase):
    def __init__(self, config, dataset, nr_proc, use_gpu=True):
        """
        Run prediction in multiprocesses, on either CPU or GPU. Mix mode not supported.

        :param nr_proc: number of processes to use
        :param use_gpu: use GPU or CPU.
            If GPU, then nr_proc cannot be larger than the total number of GPUs available
            in CUDA_VISIBLE_DEVICES or in the system.
        """
        assert config.return_input == False, "return_input not supported for MultiProcessDatasetPredictor"
        assert nr_proc > 1
        super(MultiProcessDatasetPredictor, self).__init__(config, dataset)

        self.nr_proc = nr_proc

        self.inqueue, self.inqueue_proc = dataflow_to_process_queue(
                self.dataset, nr_proc * 2, self.nr_proc)
        self.outqueue = multiprocessing.Queue()
        if use_gpu:
            try:
                gpus = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
                assert len(gpus) >= self.nr_proc, \
                        "nr_proc={} while only {} gpus available".format(
                                self.nr_proc, len(gpus))
            except KeyError:
                # TODO number of GPUs not checked
                gpus = list(range(self.nr_gpu))
        else:
            gpus = [-1] * self.nr_proc
        self.workers = [MultiProcessQueuePredictWorker(
                    i, gpus[i], self.inqueue, self.outqueue, self.config)
                        for i in range(self.nr_proc)]
        self.result_queue = OrderedResultGatherProc(
                self.outqueue, nr_producer=self.nr_proc)

        # setup all the procs
        self.inqueue_proc.start()
        for p in self.workers: p.start()
        self.result_queue.start()
        ensure_proc_terminate(self.workers + [self.result_queue, self.inqueue_proc])

    def get_result(self):
        with tqdm(total=self.dataset.size()) as pbar:
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
        self.result_queue.join()
        self.result_queue.terminate()
        for p in self.workers:
            p.join(); p.terminate()
