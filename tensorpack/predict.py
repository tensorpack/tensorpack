# -*- coding: UTF-8 -*-
# File: predict.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import tensorflow as tf
from itertools import count
import argparse
from collections import namedtuple
import numpy as np
import bisect
from tqdm import tqdm
from six.moves import zip

import multiprocessing
from .utils.concurrency import ensure_proc_terminate, OrderedResultGatherProc, DIE

from .tfutils import *
from .utils import logger
from .tfutils.modelutils import describe_model
from .dataflow import DataFlow, BatchData

__all__ = ['PredictConfig', 'DatasetPredictor', 'get_predict_func']

class PredictConfig(object):
    def __init__(self, **kwargs):
        """
        The config used by `get_predict_func`.

        :param session_config: a `tf.ConfigProto` instance to instantiate the
            session. default to a session running 1 GPU.
        :param session_init: a `utils.sessinit.SessionInit` instance to
            initialize variables of a session.
        :param input_data_mapping: Decide the mapping from each component in data
            to the input tensor, since you may not need all input variables
            of the graph to run the graph for prediction (for example
            the `label` input is not used if you only need probability
            distribution).
            It should be a list with size=len(data_point),
            where each element is an index of the input variables each
            component of the data point should be fed into.
            If not given, defaults to range(len(input_vars))

            For example, in image classification task, the testing
            dataset only provides datapoints of images (no labels). When
            the input variables of the model is: ::

                input_vars: [image_var, label_var]

            the mapping should look like: ::

                input_data_mapping: [0] # the first component in a datapoint should map to `image_var`

        :param model: a `ModelDesc` instance
        :param output_var_names: a list of names of the output variables to predict, the
            variables can be any computable tensor in the graph.
            Predict specific output might not require all input variables.
        :param nr_gpu: default to 1. Use CUDA_VISIBLE_DEVICES to control which GPU to use sepcifically.
        """
        def assert_type(v, tp):
            assert isinstance(v, tp), v.__class__
        self.session_config = kwargs.pop('session_config', get_default_sess_config())
        assert_type(self.session_config, tf.ConfigProto)
        self.session_init = kwargs.pop('session_init')
        self.model = kwargs.pop('model')
        self.input_data_mapping = kwargs.pop('input_data_mapping', None)
        self.output_var_names = kwargs.pop('output_var_names')
        self.nr_gpu = kwargs.pop('nr_gpu', 1)
        assert len(kwargs) == 0, 'Unknown arguments: {}'.format(str(kwargs.keys()))

def get_predict_func(config):
    """
    :param config: a `PredictConfig` instance.
    :returns: A prediction function that takes a list of input values, and return
        a list of output values defined in ``config.output_var_names``.
    """
    output_var_names = config.output_var_names

    # input/output variables
    input_vars = config.model.get_input_vars()
    cost_var = config.model.get_cost(input_vars, is_training=False)
    if config.input_data_mapping is None:
        input_map = input_vars
    else:
        input_map = [input_vars[k] for k in config.input_data_mapping]

    # check output_var_names against output_vars
    output_vars = [tf.get_default_graph().get_tensor_by_name(get_op_var_name(n)[1])
                   for n in output_var_names]

    sess = tf.Session(config=config.session_config)
    config.session_init.init(sess)

    def run_input(dp):
        assert len(input_map) == len(dp), \
            "Graph has {} inputs but dataset only gives {} components!".format(
                    len(input_map), len(dp))
        feed = dict(zip(input_map, dp))

        results = sess.run(output_vars, feed_dict=feed)
        if len(output_vars) == 1:
            return results[0]
        else:
            return results
    return run_input

PredictResult = namedtuple('PredictResult', ['input', 'output'])


class PredictWorker(multiprocessing.Process):
    def __init__(self, idx, gpuid, inqueue, outqueue, config):
        super(PredictWorker, self).__init__()
        self.idx = idx
        self.gpuid = gpuid
        self.inqueue = inqueue
        self.outqueue = outqueue
        self.config = config

    def run(self):
        os.environ['CUDA_VISIBLE_DEVICES'] = self.gpuid
        G = tf.Graph()     # build a graph for each process, because they don't need to share anything
        with G.as_default(), tf.device('/gpu:{}'.format(self.idx)):
            self.func = get_predict_func(self.config)
            if self.idx == 0:
                describe_model()
        while True:
            tid, dp = self.inqueue.get()
            if tid == DIE:
                self.outqueue.put((DIE, None))
                return
            else:
                res = PredictResult(dp, self.func(dp))
                self.outqueue.put((tid, res))

def DFtoQueue(ds, size, nr_consumer):
    q = multiprocessing.Queue(size)
    class EnqueProc(multiprocessing.Process):
        def __init__(self, ds, q, nr_consumer):
            super(EnqueProc, self).__init__()
            self.ds = ds
            self.q = q

        def run(self):
            for idx, dp in enumerate(self.ds.get_data()):
                self.q.put((idx, dp))
            print "Enqueue ends"
            for _ in range(nr_consumer):
                self.q.put((DIE, None))

    proc = EnqueProc(ds, q, nr_consumer)
    return q, proc

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
            self.inqueue, self.inqueue_proc = DFtoQueue(self.ds, 10, self.nr_gpu)
            self.outqueue = multiprocessing.Queue()
            try:
                gpus = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
            except KeyError:
                gpus = range(self.nr_gpu)
            self.workers = [PredictWorker(i, gpus[i], self.inqueue, self.outqueue, config)
                            for i in range(self.nr_gpu)]
            self.result_queue = OrderedResultGatherProc(self.outqueue)

            # run the procs
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
                while True:
                    res = self.result_queue.get()
                    if res[0] != DIE:
                        yield res[1]
                    else:
                        break
                    pbar.update()
                self.inqueue_proc.join()
                self.inqueue_proc.terminate()
                for p in self.workers:
                    p.join(); p.terminate()

    def get_all_result(self):
        """
        Run over the dataset and return a list of all predictions.
        """
        return list(self.get_result())

