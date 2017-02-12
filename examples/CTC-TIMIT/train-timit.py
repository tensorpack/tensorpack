#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: train-timit.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import tensorflow as tf
import numpy as np
import os
import sys
import argparse
from collections import Counter
import operator
import six
from six.moves import map, range

from tensorpack import *
from tensorpack.tfutils.gradproc import SummaryGradient, GlobalNormClip
from tensorpack.utils.globvars import globalns as param
import tensorpack.tfutils.symbolic_functions as symbf
from timitdata import TIMITBatch

BATCH = 64
NLAYER = 2
HIDDEN = 128
NR_CLASS = 61 + 1   # 61 phoneme + epsilon
FEATUREDIM = 39     # MFCC feature dimension


class Model(ModelDesc):
    def _get_inputs(self):
        return [InputDesc(tf.float32, [None, None, FEATUREDIM], 'feat'),   # bxmaxseqx39
                InputDesc(tf.int64, None, 'labelidx'),  # label is b x maxlen, sparse
                InputDesc(tf.int32, None, 'labelvalue'),
                InputDesc(tf.int64, None, 'labelshape'),
                InputDesc(tf.int32, [None], 'seqlen'),   # b
                ]

    def _build_graph(self, inputs):
        feat, labelidx, labelvalue, labelshape, seqlen = inputs
        label = tf.SparseTensor(labelidx, labelvalue, labelshape)

        cell = tf.contrib.rnn.BasicLSTMCell(num_units=HIDDEN)
        cell = tf.contrib.rnn.MultiRNNCell([cell] * NLAYER)

        initial = cell.zero_state(tf.shape(feat)[0], tf.float32)

        outputs, last_state = tf.nn.dynamic_rnn(cell, feat,
                                                seqlen, initial,
                                                dtype=tf.float32, scope='rnn')

        # o: b x t x HIDDEN
        output = tf.reshape(outputs, [-1, HIDDEN])  # (Bxt) x rnnsize
        logits = FullyConnected('fc', output, NR_CLASS, nl=tf.identity,
                                W_init=tf.truncated_normal_initializer(stddev=0.01))
        logits = tf.reshape(logits, (BATCH, -1, NR_CLASS))

        loss = tf.nn.ctc_loss(logits, label, seqlen, time_major=False)

        self.cost = tf.reduce_mean(loss, name='cost')

        logits = tf.transpose(logits, [1, 0, 2])

        isTrain = get_current_tower_context().is_training
        if isTrain:
            # beam search is too slow to run in training
            predictions = tf.to_int32(
                tf.nn.ctc_greedy_decoder(logits, seqlen)[0][0])
        else:
            predictions = tf.to_int32(
                tf.nn.ctc_beam_search_decoder(logits, seqlen)[0][0])
        err = tf.edit_distance(predictions, label, normalize=True)
        err.set_shape([None])
        err = tf.reduce_mean(err, name='error')
        summary.add_moving_summary(err, self.cost)

    def _get_optimizer(self):
        lr = symbolic_functions.get_scalar_var('learning_rate', 5e-3, summary=True)
        opt = tf.train.AdamOptimizer(lr, epsilon=1e-3)
        return optimizer.apply_grad_processors(
            opt, [GlobalNormClip(5), SummaryGradient()])


def get_data(path, isTrain, stat_file):
    ds = LMDBDataPoint(path, shuffle=isTrain)
    mean, std = serialize.loads(open(stat_file).read())
    ds = MapDataComponent(ds, lambda x: (x - mean) / std)
    ds = TIMITBatch(ds, BATCH)
    if isTrain:
        ds = PrefetchDataZMQ(ds, 1)
    return ds


def get_config(ds_train, ds_test):
    return TrainConfig(
        dataflow=ds_train,
        callbacks=[
            ModelSaver(),
            StatMonitorParamSetter('learning_rate', 'error',
                                   lambda x: x * 0.2, 0, 5),
            HumanHyperParamSetter('learning_rate'),
            PeriodicTrigger(
                InferenceRunner(ds_test, [ScalarStats('error')]),
                every_k_epochs=2),
        ],
        model=Model(),
        max_epoch=70,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--load', help='load model')
    parser.add_argument('--train', help='path to training lmdb', required=True)
    parser.add_argument('--test', help='path to testing lmdb', required=True)
    parser.add_argument('--stat', help='path to the mean/std statistics file',
                        default='stats.data')
    args = parser.parse_args()
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    logger.auto_set_dir()
    ds_train = get_data(args.train, True, args.stat)
    ds_test = get_data(args.test, False, args.stat)

    config = get_config(ds_train, ds_test)
    if args.load:
        config.session_init = SaverRestore(args.load)
    QueueInputTrainer(config).train()
