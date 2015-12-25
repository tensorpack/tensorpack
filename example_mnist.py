#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: example_mnist.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import tensorflow as tf
import numpy as np
from layers import *
from utils import *
from dataflow.dataset import Mnist
from dataflow import *

IMAGE_SIZE = 28
PIXELS = IMAGE_SIZE * IMAGE_SIZE
NUM_CLASS = 10
batch_size = 128
LOG_DIR = 'train_log'

def get_model(input, label):
    """
    Args:
        input: bxPIXELS
        label: bx1 integer
    Returns:
        (output, cost)
        output: variable
        cost: scalar variable
    """
    fc0 = FullyConnected('fc0', input, 200)
    fc0 = tf.nn.relu(fc0)
    fc1 = FullyConnected('fc1', fc0, out_dim=200)
    fc1 = tf.nn.relu(fc1)
    fc2 = FullyConnected('lr', fc1, out_dim=10)
    prob = tf.nn.softmax(fc2)

    logprob = tf.log(prob)
    y = one_hot(label, NUM_CLASS)
    cost = tf.reduce_sum(-y * logprob, 1)
    cost = tf.reduce_mean(cost, name='cost')

    tf.scalar_summary(cost.op.name, cost)
    return prob, cost

def get_eval(prob, labels):
    """
    Args:
        prob: bx10
        labels: b
    Returns:
        scalar float: accuracy
    """
    correct = tf.nn.in_top_k(prob, labels, 1)

    nr_correct = tf.reduce_sum(tf.cast(correct, tf.int32))
    return tf.cast(nr_correct, tf.float32) / tf.cast(tf.size(labels), tf.float32)

def main():
    dataset_train = BatchData(Mnist('train'), batch_size)
    dataset_test = BatchData(Mnist('test'), batch_size)
    with tf.Graph().as_default():
        input_var = tf.placeholder(tf.float32, shape=(batch_size, PIXELS))
        label_var = tf.placeholder(tf.int32, shape=(batch_size,))

        prob, cost = get_model(input_var, label_var)

        optimizer = tf.train.AdagradOptimizer(0.01)
        train_op = optimizer.minimize(cost)

        eval_op = get_eval(prob, label_var)
        summary_op = tf.merge_all_summaries()
        saver = tf.train.Saver()

        sess = tf.Session()
        init = tf.initialize_all_variables()
        sess.run(init)
        summary_writer = tf.train.SummaryWriter(LOG_DIR,
                                                graph_def=sess.graph_def)

        epoch = 0
        while True:
            epoch += 1
            for (img, label) in dataset_train.get_data():
                feed = {input_var: img,
                        label_var: label}

                _, cost_value = sess.run([train_op, cost], feed_dict=feed)

            print('Epoch %d: cost = %.2f' % (epoch, cost_value))

            summary_str = sess.run(summary_op, feed_dict=feed)
            summary_writer.add_summary(summary_str, epoch)

            if epoch % 2 == 0:
                saver.save(sess, LOG_DIR, global_step=epoch)

                scores = []
                for (img, label) in dataset_test.get_data():
                    feed = {input_var: img, label_var: label}
                    scores.append(sess.run(eval_op, feed_dict=feed))
                print "Test Scores: {}".format(np.array(scores).mean())



if __name__ == '__main__':
    main()
