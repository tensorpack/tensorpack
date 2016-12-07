#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: mnist-convnet.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import numpy as np
import tensorflow as tf
import os, sys
import argparse

"""
MNIST ConvNet example.
about 0.6% validation error after 30 epochs.
"""

# Just import everything into current namespace
from tensorpack import *

IMAGE_SIZE = 28

class Model(ModelDesc):
    def _get_input_vars(self):
        """Define all the input variables (with type, shape, name) that'll be
        fed into the graph to produce a cost.  """
        return [InputVar(tf.float32, (None, IMAGE_SIZE, IMAGE_SIZE), 'input'),
                InputVar(tf.int32, (None,), 'label') ]

    def _build_graph(self, input_vars):
        """This function should build the model which takes the input variables
        and define self.cost at the end"""

        # input_vars contains a list of input variables defined above
        image, label = input_vars
        # In tensorflow, inputs to convolution function are assumed to be
        # NHWC. Add a single channel here.
        image = tf.expand_dims(image, 3)

        image = image * 2 - 1   # center the pixels values at zero
        # The context manager `argscope` sets the default option for all the layers under
        # this context. Here we use 32 channel convolution with shape 3x3 and
        # PReLU as nonlinearity.
        with argscope(Conv2D, kernel_shape=3, nl=PReLU.f, out_channel=32):
            """
            LinearWrap is just a convenient way to compose a linear symbolic graph.
            You can also do the equivalent in tensorflow style:
            l = Conv2D('conv0', image)
            l = MaxPooling('pool0', image, 2)
            ...  """

            logits = (LinearWrap(image) # the starting brace is only for line-breaking
                    .Conv2D('conv0')
                    .MaxPooling('pool0', 2)
                    .Conv2D('conv1', padding='SAME')
                    .Conv2D('conv2')
                    .MaxPooling('pool1', 2)
                    .Conv2D('conv3')
                    .FullyConnected('fc0', 512, nl=tf.nn.relu)
                    .Dropout('dropout', 0.5)
                    .FullyConnected('fc1', out_dim=10, nl=tf.identity)())
        prob = tf.nn.softmax(logits, name='prob')   # a Bx10 with probabilities

        cost = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, label) # a vector of length B with loss of each sample
        cost = tf.reduce_mean(cost, name='cross_entropy_loss')  # the average cross-entropy loss

        # compute the "incorrect vector", for the callback ClassificationError to use at validation time
        wrong = symbolic_functions.prediction_incorrect(logits, label, name='incorrect')

        # This will monitor training error (in a moving_average fashion):
        # 1. write the value to tensosrboard
        # 2. write the value to stat.json
        # 3. print the value after each epoch
        train_error = tf.reduce_mean(wrong, name='train_error')
        summary.add_moving_summary(train_error)

        # Use a regex to find parameters to apply weight decay.
        # Here we apply a weight decay on all W (weight matrix) of all fc layers
        wd_cost = tf.mul(1e-5,
                         regularize_cost('fc.*/W', tf.nn.l2_loss),
                         name='regularize_loss')
        summary.add_moving_summary(cost, wd_cost)

        # monitor histogram of all weight (of conv and fc layers) in tensorboard
        summary.add_param_summary([('.*/W', ['histogram'])])
        self.cost = tf.add_n([wd_cost, cost], name='cost')

def get_data():
    train = BatchData(dataset.Mnist('train'), 128)
    test = BatchData(dataset.Mnist('test'), 256, remainder=True)
    return train, test

def get_config():
    # automatically setup the directory train_log/mnist-convnet for logging
    logger.auto_set_dir()

    dataset_train, dataset_test = get_data()
    # how many iterations you want in each epoch
    step_per_epoch = dataset_train.size()

    lr = tf.train.exponential_decay(
        learning_rate=1e-3,
        global_step=get_global_step_var(),
        decay_steps=dataset_train.size() * 10,
        decay_rate=0.3, staircase=True, name='learning_rate')

    # This will also put the summary in tensorboard,stat.json and print in
    # terminal, but without the moving average
    tf.summary.scalar('lr', lr)

    # get the config which contains everything necessary in a training
    return TrainConfig(
        dataset=dataset_train,  # the DataFlow instance for training
        optimizer=tf.train.AdamOptimizer(lr),
        callbacks=Callbacks([
            StatPrinter(),  # print statistics in terminal after every epoch
            ModelSaver(),   # save the model after every epoch
            InferenceRunner(    # run inference(for validation) after every epoch
                dataset_test,   # the DataFlow instance used for validation
                # Calculate both the cost and the error for this DataFlow
                [ScalarStats('cost'), ClassificationError('incorrect')]),
        ]),
        model=Model(),
        step_per_epoch=step_per_epoch,
        max_epoch=100,
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
    SimpleTrainer(config).train()

