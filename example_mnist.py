#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: example_mnist.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>


# prefer protobuf in user-namespace
import sys
import os
sys.path.insert(0, os.path.expanduser('~/.local/lib/python2.7/site-packages'))

import tensorflow as tf
import numpy as np
from itertools import count


from layers import *
from utils import *
from dataflow.dataset import Mnist
from dataflow import *

IMAGE_SIZE = 28
NUM_CLASS = 10
batch_size = 128
LOG_DIR = 'train_log'

def get_model(input, label):
    """
    Args:
        input: bx28x28
        label: bx1 integer
    Returns:
        (output, cost)
        output: variable
        cost: scalar variable
    """
    input = tf.reshape(input, [-1, IMAGE_SIZE, IMAGE_SIZE, 1])
    conv0 = Conv2D('conv0', input, out_channel=20, kernel_shape=5,
                  padding='valid')
    pool0 = tf.nn.max_pool(conv0, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                           padding='SAME')
    conv1 = Conv2D('conv1', pool0, out_channel=40, kernel_shape=3,
                  padding='valid')
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                           padding='SAME')
    conv2 = Conv2D('conv2', pool0, out_channel=40, kernel_shape=3,
                  padding='valid')

    feature = batch_flatten(conv2)

    fc0 = FullyConnected('fc0', feature, 512)
    fc0 = tf.nn.relu(fc0)
    fc1 = FullyConnected('lr', fc0, out_dim=10)
    prob = tf.nn.softmax(fc1, name='output')

    logprob = tf.log(prob)
    y = one_hot(label, NUM_CLASS)
    cost = tf.reduce_sum(-y * logprob, 1)
    cost = tf.reduce_mean(cost, name='cost')

    tf.scalar_summary(cost.op.name, cost)
    return prob, cost

def main():
    dataset_train = Mnist('train')
    dataset_test = Mnist('test')
    extensions = [
        OnehotClassificationValidation(
            BatchData(dataset_test, batch_size, remainder=True),
            prefix='test', period=2),
        PeriodicSaver(LOG_DIR, period=2)
    ]

    with tf.Graph().as_default():
        input_var = tf.placeholder(tf.float32, shape=(None, IMAGE_SIZE, IMAGE_SIZE), name='input')
        label_var = tf.placeholder(tf.int32, shape=(None,), name='label')

        prob, cost = get_model(input_var, label_var)

        optimizer = tf.train.AdagradOptimizer(0.01)
        train_op = optimizer.minimize(cost)

        for ext in extensions:
            ext.init()

        summary_op = tf.merge_all_summaries()
        config = tf.ConfigProto()
        config.device_count['GPU'] = 1
        sess = tf.Session(config=config)
        sess.run(tf.initialize_all_variables())
        summary_writer = tf.train.SummaryWriter(LOG_DIR, graph_def=sess.graph_def)

        with sess.as_default():
            for epoch in count(1):
                for (img, label) in BatchData(dataset_train, batch_size).get_data():
                    feed = {input_var: img,
                            label_var: label}

                    _, cost_value = sess.run([train_op, cost], feed_dict=feed)

                print('Epoch %d: last batch cost = %.2f' % (epoch, cost_value))
                summary_str = summary_op.eval(feed_dict=feed)
                summary_writer.add_summary(summary_str, epoch)

                for ext in extensions:
                    ext.trigger()



if __name__ == '__main__':
    main()
