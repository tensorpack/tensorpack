#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: char-rnn.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import tensorflow as tf
import numpy as np
import os, sys
import argparse
from collections import Counter
import operator

from tensorpack import *
from tensorpack.models import  *
from tensorpack.utils import  *
from tensorpack.tfutils.gradproc import  *
from tensorpack.utils.lut import LookUpTable
from tensorpack.callbacks import *

from tensorflow.models.rnn import rnn_cell
from tensorflow.models.rnn import seq2seq

BATCH_SIZE = 128
RNN_SIZE = 128  # hidden state size
NUM_RNN_LAYER = 2
SEQ_LEN = 50
VOCAB_SIZE = None   # will be initialized by CharRNNData
CORPUS = 'input.txt'

class CharRNNData(DataFlow):
    def __init__(self, input_file, size):
        self.seq_length = SEQ_LEN
        self._size = size
        self.rng = get_rng(self)

        # preprocess data
        with open(input_file) as f:
            data = f.read()
        counter = Counter(data)
        char_cnt = sorted(counter.items(), key=operator.itemgetter(1), reverse=True)
        self.chars = [x[0] for x in char_cnt]
        self.vocab_size = len(self.chars)
        global VOCAB_SIZE
        VOCAB_SIZE = self.vocab_size
        self.lut = LookUpTable(self.chars)
        self.whole_seq = np.array(list(map(self.lut.get_idx, data)), dtype='int32')

    def reset_state(self):
        self.rng = get_rng(self)

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
        return [InputVar(tf.int32, (None, SEQ_LEN), 'input'),
                InputVar(tf.int32, (None, SEQ_LEN), 'nextinput')
               ]

    def _get_cost(self, input_vars, is_training):
        input, nextinput = input_vars

        cell = rnn_cell.BasicLSTMCell(RNN_SIZE)
        cell = rnn_cell.MultiRNNCell([cell] * NUM_RNN_LAYER)

        self.initial = initial = cell.zero_state(tf.shape(input)[0], tf.float32)

        embeddingW = tf.get_variable('embedding', [VOCAB_SIZE, RNN_SIZE])
        input_feature = tf.nn.embedding_lookup(embeddingW, input) # B x seqlen x rnnsize

        input_list = tf.split(1, SEQ_LEN, input_feature)    #seqlen x (Bx1xrnnsize)
        input_list = [tf.squeeze(x, [1]) for x in input_list]

        # seqlen is 1 in inference. don't need loop_function
        outputs, last_state = seq2seq.rnn_decoder(input_list, initial, cell, scope='rnnlm')
        self.last_state = tf.identity(last_state, 'last_state')
        # seqlen x (Bxrnnsize)
        output = tf.reshape(tf.concat(1, outputs), [-1, RNN_SIZE])  # (seqlenxB) x rnnsize
        logits = FullyConnected('fc', output, VOCAB_SIZE, nl=tf.identity)
        self.prob = tf.nn.softmax(logits)

        xent_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits, symbolic_functions.flatten(nextinput))
        xent_loss = tf.reduce_mean(xent_loss, name='xent_loss')
        tf.add_to_collection(MOVING_SUMMARY_VARS_KEY, xent_loss)

        summary.add_param_summary([('.*/W', ['histogram'])])   # monitor histogram of all W
        return tf.add_n([xent_loss], name='cost')

    def get_gradient_processor(self):
        return [MapGradient(lambda grad: tf.clip_by_global_norm([grad], 5.)[0][0])]

def get_config():
    basename = os.path.basename(__file__)
    logger.set_logger_dir(
        os.path.join('train_log', basename[:basename.rfind('.')]))

    ds = CharRNNData(CORPUS, 100000)
    ds = BatchData(ds, 128)
    step_per_epoch = ds.size()

    lr = tf.Variable(2e-3, trainable=False, name='learning_rate')
    tf.scalar_summary('learning_rate', lr)

    return TrainConfig(
        dataset=ds,
        optimizer=tf.train.AdamOptimizer(lr),
        callbacks=Callbacks([
            StatPrinter(),
            ModelSaver(),
            HumanHyperParamSetter('learning_rate', 'hyper.txt')
        ]),
        model=Model(),
        step_per_epoch=step_per_epoch,
        max_epoch=50,
    )

def sample(path, start, length):
    """
    :param path: path to the model
    :param start: a `str`. the starting characters
    :param length: a `int`. the length of text to generate
    """
    # initialize vocabulary and sequence length
    global SEQ_LEN
    SEQ_LEN = 1
    ds = CharRNNData(CORPUS, 100000)

    model = Model()
    input_vars = model.get_input_vars()
    model.get_cost(input_vars, False)
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
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.') # nargs='*' in multi mode
    parser.add_argument('--load', help='load model')
    subparsers = parser.add_subparsers(title='command', dest='command')
    parser_sample = subparsers.add_parser('sample', help='sample a trained model')
    parser_sample.add_argument('-n', '--num', type=int, default=300,
            help='length of text to generate')
    parser_sample.add_argument('-s', '--start', required=True, default='The ',
            help='initial text sequence')
    args = parser.parse_args()
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    if args.command == 'sample':
        sample(args.load, args.start, args.num)
        sys.exit()
    else:
        with tf.Graph().as_default():
            config = get_config()
            if args.load:
                config.session_init = SaverRestore(args.load)
            QueueInputTrainer(config).train()

