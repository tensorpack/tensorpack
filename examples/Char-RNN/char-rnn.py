#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: char-rnn.py
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
from tensorpack.tfutils.gradproc import GlobalNormClip
from tensorpack.utils.lut import LookUpTable
from tensorpack.utils.globvars import globalns as param
rnn = tf.contrib.rnn

# some model hyperparams to set
param.batch_size = 128
param.rnn_size = 256
param.num_rnn_layer = 2
param.seq_len = 50
param.grad_clip = 5.
param.vocab_size = None
param.softmax_temprature = 1
param.corpus = 'input.txt'


class CharRNNData(RNGDataFlow):
    def __init__(self, input_file, size):
        self.seq_length = param.seq_len
        self._size = size
        self.rng = get_rng(self)

        logger.info("Loading corpus...")
        # preprocess data
        with open(input_file, 'rb') as f:
            data = f.read()
        if six.PY2:
            data = bytearray(data)
        data = [chr(c) for c in data if c < 128]
        counter = Counter(data)
        char_cnt = sorted(counter.items(), key=operator.itemgetter(1), reverse=True)
        self.chars = [x[0] for x in char_cnt]
        print(sorted(self.chars))
        self.vocab_size = len(self.chars)
        param.vocab_size = self.vocab_size
        self.lut = LookUpTable(self.chars)
        self.whole_seq = np.array(list(map(self.lut.get_idx, data)), dtype='int32')
        logger.info("Corpus loaded. Vocab size: {}".format(self.vocab_size))

    def size(self):
        return self._size

    def get_data(self):
        random_starts = self.rng.randint(0,
                                         self.whole_seq.shape[0] - self.seq_length - 1, (self._size,))
        for st in random_starts:
            seq = self.whole_seq[st:st + self.seq_length + 1]
            yield [seq[:-1], seq[1:]]


class Model(ModelDesc):
    def _get_inputs(self):
        return [InputDesc(tf.int32, (None, param.seq_len), 'input'),
                InputDesc(tf.int32, (None, param.seq_len), 'nextinput')]

    def _build_graph(self, inputs):
        input, nextinput = inputs

        cell = rnn.BasicLSTMCell(num_units=param.rnn_size)
        cell = rnn.MultiRNNCell([cell] * param.num_rnn_layer)

        def get_v(n):
            ret = tf.get_variable(n + '_unused', [param.batch_size, param.rnn_size],
                                  trainable=False,
                                  initializer=tf.constant_initializer())
            ret = symbolic_functions.shapeless_placeholder(ret, 0, name=n)
            return ret
        self.initial = initial = \
            (rnn.LSTMStateTuple(get_v('c0'), get_v('h0')),
             rnn.LSTMStateTuple(get_v('c1'), get_v('h1')))

        embeddingW = tf.get_variable('embedding', [param.vocab_size, param.rnn_size])
        input_feature = tf.nn.embedding_lookup(embeddingW, input)  # B x seqlen x rnnsize

        input_list = tf.unstack(input_feature, axis=1)  # seqlen x (Bxrnnsize)

        # seqlen is 1 in inference. don't need loop_function
        outputs, last_state = rnn.static_rnn(cell, input_list, initial, scope='rnnlm')
        self.last_state = tf.identity(last_state, 'last_state')

        # seqlen x (Bxrnnsize)
        output = tf.reshape(tf.concat(outputs, 1), [-1, param.rnn_size])  # (Bxseqlen) x rnnsize
        logits = FullyConnected('fc', output, param.vocab_size, nl=tf.identity)
        self.prob = tf.nn.softmax(logits / param.softmax_temprature, name='prob')

        xent_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=tf.reshape(nextinput, [-1]))
        self.cost = tf.reduce_mean(xent_loss, name='cost')
        summary.add_param_summary(('.*/W', ['histogram']))   # monitor histogram of all W
        summary.add_moving_summary(self.cost)

    def _get_optimizer(self):
        lr = symbolic_functions.get_scalar_var('learning_rate', 2e-3, summary=True)
        opt = tf.train.AdamOptimizer(lr)
        return optimizer.apply_grad_processors(opt, [GlobalNormClip(5)])


def get_config():
    logger.auto_set_dir()

    ds = CharRNNData(param.corpus, 100000)
    ds = BatchData(ds, param.batch_size)

    return TrainConfig(
        dataflow=ds,
        callbacks=[
            ModelSaver(),
            ScheduledHyperParamSetter('learning_rate', [(25, 2e-4)])
        ],
        model=Model(),
        max_epoch=50,
    )


def sample(path, start, length):
    """
    :param path: path to the model
    :param start: a `str`. the starting characters
    :param length: a `int`. the length of text to generate
    """
    # initialize vocabulary and sequence length
    param.seq_len = 1
    ds = CharRNNData(param.corpus, 100000)

    pred = OfflinePredictor(PredictConfig(
        model=Model(),
        session_init=SaverRestore(path),
        input_names=['input', 'c0', 'h0', 'c1', 'h1'],
        output_names=['prob', 'last_state']))

    # feed the starting sentence
    initial = np.zeros((1, param.rnn_size))
    for c in start[:-1]:
        x = np.array([[ds.lut.get_idx(c)]], dtype='int32')
        _, state = pred(x, initial, initial, initial, initial)

    def pick(prob):
        t = np.cumsum(prob)
        s = np.sum(prob)
        return(int(np.searchsorted(t, np.random.rand(1) * s)))

    # generate more
    ret = start
    c = start[-1]
    for k in range(length):
        x = np.array([[ds.lut.get_idx(c)]], dtype='int32')
        prob, state = pred(x, state[0, 0], state[0, 1], state[1, 0], state[1, 1])
        c = ds.lut.get_obj(pick(prob[0]))
        ret += c
    print(ret)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--load', help='load model')
    subparsers = parser.add_subparsers(title='command', dest='command')
    parser_sample = subparsers.add_parser('sample', help='sample a trained model')
    parser_sample.add_argument('-n', '--num', type=int,
                               default=300, help='length of text to generate')
    parser_sample.add_argument('-s', '--start',
                               default='The ', help='initial text sequence')
    parser_sample.add_argument('-t', '--temperature', type=float,
                               default=1, help='softmax temperature')
    parser_train = subparsers.add_parser('train', help='train')
    args = parser.parse_args()
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if args.command == 'sample':
        param.softmax_temprature = args.temperature
        assert args.load is not None, "Load your model by argument --load"
        sample(args.load, args.start, args.num)
        sys.exit()
    else:
        config = get_config()
        if args.load:
            config.session_init = SaverRestore(args.load)
        QueueInputTrainer(config).train()
