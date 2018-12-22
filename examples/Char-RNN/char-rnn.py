#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: char-rnn.py
# Author: Yuxin Wu

import argparse
import numpy as np
import operator
import os
import sys
from collections import Counter
import six
import tensorflow as tf
from six.moves import range

from tensorpack import *
from tensorpack.tfutils import optimizer, summary
from tensorpack.tfutils.gradproc import GlobalNormClip

rnn = tf.contrib.rnn

class _NS: pass  # noqa


param = _NS()

# some model hyperparams to set
param.batch_size = 128
param.rnn_size = 256
param.num_rnn_layer = 2
param.seq_len = 50
param.grad_clip = 5.
param.vocab_size = None
param.softmax_temprature = 1
param.corpus = None


class CharRNNData(RNGDataFlow):
    def __init__(self, input_file, size):
        self.seq_length = param.seq_len
        self._size = size

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
        self.char2idx = {c: i for i, c in enumerate(self.chars)}
        self.whole_seq = np.array([self.char2idx[c] for c in data], dtype='int32')
        logger.info("Corpus loaded. Vocab size: {}".format(self.vocab_size))

    def __len__(self):
        return self._size

    def __iter__(self):
        random_starts = self.rng.randint(
            0, self.whole_seq.shape[0] - self.seq_length - 1, (self._size,))
        for st in random_starts:
            seq = self.whole_seq[st:st + self.seq_length + 1]
            yield [seq[:-1], seq[1:]]


class Model(ModelDesc):
    def inputs(self):
        return [tf.placeholder(tf.int32, (None, param.seq_len), 'input'),
                tf.placeholder(tf.int32, (None, param.seq_len), 'nextinput')]

    def build_graph(self, input, nextinput):
        cell = rnn.MultiRNNCell([rnn.LSTMBlockCell(num_units=param.rnn_size)
                                for _ in range(param.num_rnn_layer)])

        def get_v(n):
            ret = tf.get_variable(n + '_unused', [param.batch_size, param.rnn_size],
                                  trainable=False,
                                  initializer=tf.constant_initializer())
            ret = tf.placeholder_with_default(ret, shape=[None, param.rnn_size], name=n)
            return ret
        initial = (rnn.LSTMStateTuple(get_v('c0'), get_v('h0')),
                   rnn.LSTMStateTuple(get_v('c1'), get_v('h1')))

        embeddingW = tf.get_variable('embedding', [param.vocab_size, param.rnn_size])
        input_feature = tf.nn.embedding_lookup(embeddingW, input)  # B x seqlen x rnnsize

        input_list = tf.unstack(input_feature, axis=1)  # seqlen x (Bxrnnsize)

        outputs, last_state = rnn.static_rnn(cell, input_list, initial, scope='rnnlm')
        last_state = tf.identity(last_state, 'last_state')

        # seqlen x (Bxrnnsize)
        output = tf.reshape(tf.concat(outputs, 1), [-1, param.rnn_size])  # (Bxseqlen) x rnnsize
        logits = FullyConnected('fc', output, param.vocab_size, activation=tf.identity)
        tf.nn.softmax(logits / param.softmax_temprature, name='prob')

        xent_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=tf.reshape(nextinput, [-1]))
        cost = tf.reduce_mean(xent_loss, name='cost')
        summary.add_param_summary(('.*/W', ['histogram']))   # monitor histogram of all W
        summary.add_moving_summary(cost)
        return cost

    def optimizer(self):
        lr = tf.get_variable('learning_rate', initializer=2e-3, trainable=False)
        opt = tf.train.AdamOptimizer(lr)
        return optimizer.apply_grad_processors(opt, [GlobalNormClip(5)])


def get_config():
    logger.auto_set_dir()

    ds = CharRNNData(param.corpus, 100000)
    ds = BatchData(ds, param.batch_size)

    return TrainConfig(
        data=QueueInput(ds),
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
        x = np.array([[ds.char2idx[c]]], dtype='int32')
        _, state = pred(x, initial, initial, initial, initial)

    def pick(prob):
        t = np.cumsum(prob)
        s = np.sum(prob)
        return(int(np.searchsorted(t, np.random.rand(1) * s)))

    # generate more
    ret = start
    c = start[-1]
    for k in range(length):
        x = np.array([[ds.char2idx[c]]], dtype='int32')
        prob, state = pred(x, state[0, 0], state[0, 1], state[1, 0], state[1, 1])
        c = ds.chars[pick(prob[0])]
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
    parser_train.add_argument('--corpus', help='corpus file', default='input.txt')
    args = parser.parse_args()
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if args.command == 'sample':
        param.softmax_temprature = args.temperature
        assert args.load is not None, "Load your model by argument --load"
        sample(args.load, args.start, args.num)
        sys.exit()
    else:
        param.corpus = args.corpus
        config = get_config()
        if args.load:
            config.session_init = SaverRestore(args.load)
        launch_train_with_config(config, SimpleTrainer())
