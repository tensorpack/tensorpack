#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: PTB-LSTM.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import tensorflow as tf
import numpy as np
import os
import argparse

from tensorpack import *
from tensorpack.tfutils.gradproc import *
from tensorpack.utils import logger
from tensorpack.utils.fs import download, get_dataset_path
from tensorpack.utils.argtools import memoized_ignoreargs

import reader as tfreader
from reader import ptb_producer
rnn = tf.contrib.rnn

SEQ_LEN = 35
HIDDEN_SIZE = 650
NUM_LAYER = 2
BATCH = 20
DROPOUT = 0.5
VOCAB_SIZE = None
TRAIN_URL = 'https://raw.githubusercontent.com/tomsercu/lstm/master/data/ptb.train.txt'
VALID_URL = 'https://raw.githubusercontent.com/tomsercu/lstm/master/data/ptb.valid.txt'
TEST_URL = 'https://raw.githubusercontent.com/tomsercu/lstm/master/data/ptb.test.txt'


@memoized_ignoreargs
def get_PennTreeBank(data_dir=None):
    if data_dir is None:
        data_dir = get_dataset_path('ptb_data')
    if not os.path.isfile(os.path.join(data_dir, 'ptb.train.txt')):
        download(TRAIN_URL, data_dir)
        download(VALID_URL, data_dir)
        download(TEST_URL, data_dir)
    word_to_id = tfreader._build_vocab(os.path.join(data_dir, 'ptb.train.txt'))
    data3 = [np.asarray(tfreader._file_to_word_ids(os.path.join(data_dir, fname), word_to_id))
             for fname in ['ptb.train.txt', 'ptb.valid.txt', 'ptb.test.txt']]
    return data3, word_to_id


class Model(ModelDesc):
    def _get_inputs(self):
        return [InputDesc(tf.int32, (None, SEQ_LEN), 'input'),
                InputDesc(tf.int32, (None, SEQ_LEN), 'nextinput')]

    def _build_graph(self, inputs):
        is_training = get_current_tower_context().is_training
        input, nextinput = inputs
        initializer = tf.random_uniform_initializer(-0.05, 0.05)

        cell = rnn.BasicLSTMCell(num_units=HIDDEN_SIZE, forget_bias=0.0)
        if is_training:
            cell = rnn.DropoutWrapper(cell, output_keep_prob=DROPOUT)
        cell = rnn.MultiRNNCell([cell] * NUM_LAYER)

        def get_v(n):
            return tf.get_variable(n, [BATCH, HIDDEN_SIZE],
                                   trainable=False,
                                   initializer=tf.constant_initializer())
        self.state = state_var = \
            (rnn.LSTMStateTuple(get_v('c0'), get_v('h0')),
             rnn.LSTMStateTuple(get_v('c1'), get_v('h1')))

        embeddingW = tf.get_variable('embedding', [VOCAB_SIZE, HIDDEN_SIZE], initializer=initializer)
        input_feature = tf.nn.embedding_lookup(embeddingW, input)  # B x seqlen x hiddensize
        input_feature = Dropout(input_feature, DROPOUT)

        with tf.variable_scope('LSTM', initializer=initializer):
            input_list = tf.unstack(input_feature, num=SEQ_LEN, axis=1)  # seqlen x (Bxhidden)
            outputs, last_state = rnn.static_rnn(cell, input_list, state_var, scope='rnn')

        # update the hidden state after a rnn loop completes
        update_state_ops = [
            tf.assign(state_var[0].c, last_state[0].c),
            tf.assign(state_var[0].h, last_state[0].h),
            tf.assign(state_var[1].c, last_state[1].c),
            tf.assign(state_var[1].h, last_state[1].h)]

        # seqlen x (Bxrnnsize)
        output = tf.reshape(tf.concat(outputs, 1), [-1, HIDDEN_SIZE])  # (Bxseqlen) x hidden
        logits = FullyConnected('fc', output, VOCAB_SIZE, nl=tf.identity, W_init=initializer, b_init=initializer)
        xent_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=tf.reshape(nextinput, [-1]))

        with tf.control_dependencies(update_state_ops):
            self.cost = tf.truediv(tf.reduce_sum(xent_loss),
                                   tf.cast(BATCH, tf.float32), name='cost')  # log-perplexity

        perpl = tf.exp(self.cost / SEQ_LEN, name='perplexity')
        summary.add_moving_summary(perpl, self.cost)

    def reset_lstm_state(self):
        s = self.state
        z = tf.zeros_like(s[0].c)
        return tf.group(s[0].c.assign(z),
                        s[0].h.assign(z),
                        s[1].c.assign(z),
                        s[1].h.assign(z), name='reset_lstm_state')

    def _get_optimizer(self):
        lr = symbolic_functions.get_scalar_var('learning_rate', 1, summary=True)
        opt = tf.train.GradientDescentOptimizer(lr)
        return optimizer.apply_grad_processors(
            opt, [gradproc.GlobalNormClip(5)])


def get_config():
    logger.auto_set_dir()

    data3, wd2id = get_PennTreeBank()
    global VOCAB_SIZE
    VOCAB_SIZE = len(wd2id)
    steps_per_epoch = (data3[0].shape[0] // BATCH - 1) // SEQ_LEN

    train_data = TensorInput(
        lambda: ptb_producer(data3[0], BATCH, SEQ_LEN),
        steps_per_epoch)
    val_data = TensorInput(
        lambda: ptb_producer(data3[1], BATCH, SEQ_LEN),
        (data3[1].shape[0] // BATCH - 1) // SEQ_LEN)

    test_data = TensorInput(
        lambda: ptb_producer(data3[2], BATCH, SEQ_LEN),
        (data3[2].shape[0] // BATCH - 1) // SEQ_LEN)

    M = Model()
    return TrainConfig(
        data=train_data,
        model=M,
        callbacks=[
            ModelSaver(),
            HyperParamSetterWithFunc(
                'learning_rate',
                lambda e, x: x * 0.80 if e > 6 else x),
            RunOp(lambda: M.reset_lstm_state()),
            FeedfreeInferenceRunner(val_data, [ScalarStats(['cost'])]),
            RunOp(lambda: M.reset_lstm_state()),
            FeedfreeInferenceRunner(
                test_data,
                [ScalarStats(['cost'], prefix='test')], prefix='test'),
            RunOp(lambda: M.reset_lstm_state()),
            CallbackFactory(
                trigger_epoch=lambda self:
                [self.trainer.monitors.put(
                    'validation_perplexity',
                    np.exp(self.trainer.monitors.get_latest('validation_cost') / SEQ_LEN)),
                 self.trainer.monitors.put(
                     'test_perplexity',
                     np.exp(self.trainer.monitors.get_latest('test_cost') / SEQ_LEN))]
            ),
        ],
        max_epoch=70,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--load', help='load model')
    args = parser.parse_args()
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    config = get_config()
    if args.load:
        config.session_init = SaverRestore(args.load)
    SimpleFeedfreeTrainer(config).train()
