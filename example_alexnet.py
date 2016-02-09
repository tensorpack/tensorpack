#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: example_alexnet.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import tensorflow as tf
import numpy as np
import os
import argparse
import cPickle as pkl

from tensorpack.train import TrainConfig, start_train
from tensorpack.predict import PredictConfig, get_predict_func
from tensorpack.models import *
from tensorpack.utils import *
from tensorpack.utils.symbolic_functions import *
from tensorpack.utils.summary import *
from tensorpack.callbacks import *
from tensorpack.dataflow import *

BATCH_SIZE = 10
MIN_AFTER_DEQUEUE = 500
CAPACITY = MIN_AFTER_DEQUEUE + 3 * BATCH_SIZE

class Model(ModelDesc):
    def _get_input_vars(self):
        return [
            tf.placeholder(
                tf.float32, shape=(None, 227, 227, 3), name='input'),
            tf.placeholder(
                tf.int32, shape=(None,), name='label')
        ]

    def _get_cost(self, inputs, is_training):
        # img: 227x227x3
        is_training = bool(is_training)
        keep_prob = tf.constant(0.5 if is_training else 1.0)

        image, label = inputs

        l = Conv2D('conv1', image, out_channel=96, kernel_shape=11, stride=4, padding='VALID')
        l = tf.nn.lrn(l, 2, bias=1.0, alpha=2e-5, beta=0.75, name='norm1')
        l = MaxPooling('pool1', l, 3, stride=2, padding='VALID')

        l = Conv2D('conv2', l, out_channel=256, kernel_shape=5,
                       padding='SAME', split=2)
        l = tf.nn.lrn(l, 2, bias=1.0, alpha=2e-5, beta=0.75, name='norm2')
        l = MaxPooling('pool2', l, 3, stride=2, padding='VALID')

        l = Conv2D('conv3', l, out_channel=384, kernel_shape=3,
                       padding='SAME')
        l = Conv2D('conv4', l, out_channel=384, kernel_shape=3,
                       padding='SAME', split=2)
        l = Conv2D('conv5', l, out_channel=256, kernel_shape=3,
                       padding='SAME', split=2)
        l = MaxPooling('pool3', l, 3, stride=2, padding='VALID')

        l = FullyConnected('fc6', l, 4096)
        l = FullyConnected('fc7', l, out_dim=4096)
        # fc will have activation summary by default. disable this for the output layer
        logits = FullyConnected('fc8', l, out_dim=1000, summary_activation=False, nl=tf.identity)
        prob = tf.nn.softmax(logits, name='output')

        y = one_hot(label, 1000)
        cost = tf.nn.softmax_cross_entropy_with_logits(logits, y)
        cost = tf.reduce_mean(cost, name='cross_entropy_loss')
        tf.add_to_collection(MOVING_SUMMARY_VARS_KEY, cost)

        # compute the number of failed samples, for ValidationError to use at test time
        wrong = tf.not_equal(
            tf.cast(tf.argmax(prob, 1), tf.int32), label)
        wrong = tf.cast(wrong, tf.float32)
        nr_wrong = tf.reduce_sum(wrong, name='wrong')
        # monitor training error
        tf.add_to_collection(
            MOVING_SUMMARY_VARS_KEY, tf.reduce_mean(wrong, name='train_error'))

        # weight decay on all W of fc layers
        wd_cost = tf.mul(1e-4,
                         regularize_cost('fc.*/W', tf.nn.l2_loss),
                         name='regularize_loss')
        tf.add_to_collection(MOVING_SUMMARY_VARS_KEY, wd_cost)

        add_param_summary('.*/W')   # monitor histogram of all W
        return tf.add_n([wd_cost, cost], name='cost')

def get_config():
    basename = os.path.basename(__file__)
    log_dir = os.path.join('train_log', basename[:basename.rfind('.')])
    logger.set_logger_file(os.path.join(log_dir, 'training.log'))

    dataset_train = FakeData([(227,227,3), tuple()], 10)
    dataset_train = BatchData(dataset_train, 10)
    step_per_epoch = 1

    sess_config = get_default_sess_config()
    sess_config.gpu_options.per_process_gpu_memory_fraction = 0.5

    lr = tf.train.exponential_decay(
        learning_rate=1e-8,
        global_step=get_global_step_var(),
        decay_steps=dataset_train.size() * 50,
        decay_rate=0.1, staircase=True, name='learning_rate')
    tf.scalar_summary('learning_rate', lr)

    param_dict = np.load('alexnet.npy').item()

    return TrainConfig(
        dataset=dataset_train,
        optimizer=tf.train.AdamOptimizer(lr),
        callbacks=Callbacks([
            SummaryWriter(),
            PeriodicSaver(),
            #ValidationError(dataset_test, prefix='test'),
        ]),
        session_config=sess_config,
        model=Model(),
        step_per_epoch=step_per_epoch,
        session_init=ParamRestore(param_dict),
        max_epoch=100,
    )

def run_test(path):
    param_dict = np.load(path).item()

    pred_config = PredictConfig(
        model=Model(),
        input_data_mapping=[0],
        session_init=ParamRestore(param_dict),
        output_var_names=['output:0']   # output:0 is the probability distribution
    )
    predict_func = get_predict_func(pred_config)

    import cv2
    im = cv2.imread('cat.jpg')
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = cv2.resize(im, (227, 227))
    im = np.reshape(im, (1, 227, 227, 3))
    outputs = predict_func([im])[0]
    prob = outputs[0]
    print prob.shape
    ret = prob.argsort()[-10:][::-1]
    print ret
    assert ret[0] == 285

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.') # nargs='*' in multi mode
    args = parser.parse_args()
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    #start_train(get_config())

    # run alexnet with given model (in npy format)
    run_test('alexnet.npy')
