#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: char-rnn.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import tensorflow as tf
import numpy as np
import os, sys
import argparse
from collections import Counter
import operator
import six
from six.moves import map, range

from tensorpack import *
from tensorpack.tfutils.gradproc import  *
from tensorpack.utils.lut import LookUpTable
from tensorpack.utils.globvars import globalns as param

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
        with open(input_file) as f:
            data = f.read()
        counter = Counter(data)
        char_cnt = sorted(counter.items(), key=operator.itemgetter(1), reverse=True)
        self.chars = [x[0] for x in char_cnt]
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
    def _get_input_vars(self):
        return [InputVar(tf.int32, (None, param.seq_len), 'input'),
                InputVar(tf.int32, (None, param.seq_len), 'nextinput') ]

    def _build_graph(self, input_vars):
        input, nextinput = input_vars

        cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=param.rnn_size)
        cell = tf.nn.rnn_cell.MultiRNNCell([cell] * param.num_rnn_layer)

        self.initial = initial = cell.zero_state(tf.shape(input)[0], tf.float32)

        embeddingW = tf.get_variable('embedding', [param.vocab_size, param.rnn_size])
        input_feature = tf.nn.embedding_lookup(embeddingW, input) # B x seqlen x rnnsize

        input_list = tf.unstack(input_feature, axis=1)    #seqlen x (Bxrnnsize)

        # seqlen is 1 in inference. don't need loop_function
        outputs, last_state = tf.nn.rnn(cell, input_list, initial, scope='rnnlm')
        self.last_state = tf.identity(last_state, 'last_state')

        # seqlen x (Bxrnnsize)
        output = tf.reshape(tf.concat(1, outputs), [-1, param.rnn_size])  # (Bxseqlen) x rnnsize
        logits = FullyConnected('fc', output, param.vocab_size, nl=tf.identity)
        self.prob = tf.nn.softmax(logits / param.softmax_temprature)

        xent_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits, symbolic_functions.flatten(nextinput))
        self.cost = tf.reduce_mean(xent_loss, name='cost')
        summary.add_param_summary([('.*/W', ['histogram'])])   # monitor histogram of all W

    def get_gradient_processor(self):
        return [GlobalNormClip(5)]

def get_config():
    logger.auto_set_dir()

    ds = CharRNNData(param.corpus, 100000)
    ds = BatchData(ds, param.batch_size)
    step_per_epoch = ds.size()

    lr = symbolic_functions.get_scalar_var('learning_rate', 2e-3, summary=True)

    return TrainConfig(
        dataset=ds,
        optimizer=tf.train.AdamOptimizer(lr),
        callbacks=Callbacks([
            StatPrinter(), ModelSaver(),
            ScheduledHyperParamSetter('learning_rate', [(25, 2e-4)])
        ]),
        model=Model(),
        step_per_epoch=step_per_epoch,
        max_epoch=50,
    )

# TODO rewrite using Predictor interface
def sample(path, start, length):
    """
    :param path: path to the model
    :param start: a `str`. the starting characters
    :param length: a `int`. the length of text to generate
    """
    # initialize vocabulary and sequence length
    param.seq_len = 1
    ds = CharRNNData(param.corpus, 100000)

    model = Model()
    input_vars = model.get_input_vars()
    model.build_graph(input_vars, False)
    sess = tf.Session()
    tfutils.SaverRestore(path).init(sess)

    dummy_input = np.zeros((1,1), dtype='int32')
    with sess.as_default():
        # feed the starting sentence
        state = model.initial.eval({input_vars[0]: dummy_input})
        for c in start[:-1]:
            x = np.array([[ds.lut.get_idx(c)]], dtype='int32')
            state = model.last_state.eval({input_vars[0]: x, model.initial: state})

        def pick(prob):
            t = np.cumsum(prob)
            s = np.sum(prob)
            return(int(np.searchsorted(t, np.random.rand(1) * s)))

        # generate more
        ret = start
        c = start[-1]
        for k in range(length):
            x = np.array([[ds.lut.get_idx(c)]], dtype='int32')
            [prob, state] = sess.run([model.prob, model.last_state],
                    {input_vars[0]: x, model.initial: state})
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

