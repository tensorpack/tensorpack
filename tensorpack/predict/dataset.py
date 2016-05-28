#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: dataset.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

from six.moves import range
from tqdm import tqdm

from ..dataflow import DataFlow, BatchData
from ..dataflow.dftools import dataflow_to_process_queue
from ..utils.concurrency import ensure_proc_terminate, OrderedResultGatherProc, DIE

from .concurrency import *

__all__ = ['DatasetPredictor']

class DatasetPredictor(object):
    """
    Run the predict_config on a given `DataFlow`.
    """
    def __init__(self, config, dataset):
        """
        :param config: a `PredictConfig` instance.
        :param dataset: a `DataFlow` instance.
        """
        assert isinstance(dataset, DataFlow)
        self.ds = dataset
        self.nr_gpu = config.nr_gpu
        if self.nr_gpu > 1:
            self.inqueue, self.inqueue_proc = dataflow_to_process_queue(self.ds, 10, self.nr_gpu)
            self.outqueue = multiprocessing.Queue()
            try:
                gpus = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
            except KeyError:
                gpus = list(range(self.nr_gpu))
            self.workers = [QueuePredictWorker(i, gpus[i], self.inqueue, self.outqueue, config)
                            for i in range(self.nr_gpu)]
            self.result_queue = OrderedResultGatherProc(self.outqueue)

            # setup all the procs
            self.inqueue_proc.start()
            for p in self.workers: p.start()
            self.result_queue.start()
            ensure_proc_terminate(self.workers)
            ensure_proc_terminate([self.result_queue, self.inqueue_proc])
        else:
            self.func = get_predict_func(config)

    def get_result(self):
        """ A generator to produce prediction for each data"""
        with tqdm(total=self.ds.size()) as pbar:
            if self.nr_gpu == 1:
                for dp in self.ds.get_data():
                    yield PredictResult(dp, self.func(dp))
                    pbar.update()
            else:
                die_cnt = 0
                while True:
                    res = self.result_queue.get()
                    pbar.update()
                    if res[0] != DIE:
                        yield res[1]
                    else:
                        die_cnt += 1
                        if die_cnt == self.nr_gpu:
                            break
                self.inqueue_proc.join()
                self.inqueue_proc.terminate()
                for p in self.workers:
                    p.join(); p.terminate()

    def get_all_result(self):
        """
        Run over the dataset and return a list of all predictions.
        """
        return list(self.get_result())
