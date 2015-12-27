#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: validation_callback.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import tensorflow as tf
from .stat import *
from .callback import PeriodicCallback, Callback
from .naming import *
from .utils import *

class ValidationAccuracy(PeriodicCallback):
    """
    Validate the accuracy for the given correct and cost variable
    Use under the following setup:
        correct_var: integer, number of correct samples in this batch
        ds: batched dataset
    """
    def __init__(self, ds, prefix,
                 period=1,
                 correct_var_name='correct:0',
                 cost_var_name='cost:0'):
        super(ValidationAccuracy, self).__init__(period)
        self.ds = ds
        self.prefix = prefix

        self.correct_var_name = correct_var_name
        self.cost_var_name = cost_var_name

    def get_tensor(self, name):
        return self.graph.get_tensor_by_name(name)

    def _before_train(self):
        self.input_vars = tf.get_collection(INPUT_VARS_KEY)
        self.dropout_var = self.get_tensor(DROPOUT_PROB_VAR_NAME)
        self.correct_var = self.get_tensor(self.correct_var_name)
        self.cost_var = self.get_tensor(self.cost_var_name)
        self.writer = tf.get_collection(SUMMARY_WRITER_COLLECTION_KEY)[0]

    def _trigger(self):
        cnt = 0
        correct_stat = Accuracy()
        cost_sum = 0
        for dp in self.ds.get_data():
            feed = {self.dropout_var: 1.0}
            feed.update(dict(zip(self.input_vars, dp)))

            batch_size = dp[0].shape[0]   # assume batched input

            cnt += batch_size
            correct, cost = self.sess.run(
                [self.correct_var, self.cost_var], feed_dict=feed)
            correct_stat.feed(correct, batch_size)
            # each batch might not have the same size in validation
            cost_sum += cost * batch_size

        cost_avg = cost_sum / cnt
        self.writer.add_summary(
            create_summary('{}_accuracy'.format(self.prefix),
                           correct_stat.accuracy),
            self.epoch_num)
        self.writer.add_summary(
            create_summary('{}_cost'.format(self.prefix),
                           cost_avg),
            self.epoch_num)
        print "{} validation after epoch {}: acc={}, cost={}".format(
            self.prefix, self.epoch_num, correct_stat.accuracy, cost_avg)

# use SUMMARY_VARIABLES instead
#class TrainingAccuracy(Callback):
    #"""
    #Record the accuracy and cost during each step of trianing.
    #The result is a running average, thus not directly comparable with ValidationAccuracy
    #"""
    #def __init__(self, batch_size, correct_var_name='correct:0'):
        #"""
            #correct_var: number of correct sample in this batch
        #"""
        #self.correct_var_name = correct_var_name
        #self.batch_size = batch_size
        #self.epoch_num = 0

    #def _before_train(self):
        #self.writer = tf.get_collection(SUMMARY_WRITER_COLLECTION_KEY)[0]
        #output_vars = tf.get_collection(OUTPUT_VARS_KEY)
        #for idx, var in enumerate(output_vars):
            #if var.name == self.correct_var_name:
                #self.correct_output_idx = idx
                #break
        #else:
            #raise RuntimeError(
                #"'correct' variable must be in the model outputs to use TrainingAccuracy")
        #self.running_cost = StatCounter()
        #self.running_acc = Accuracy()

    #def trigger_step(self, inputs, outputs, cost):
        #self.running_cost.feed(cost)
        #self.running_acc.feed(
            #outputs[self.correct_output_idx],
            #self.batch_size) # assume batch input

    #def trigger_epoch(self):
        #self.epoch_num += 1
        #print('Training average in Epoch {}: cost={}, acc={}'.format
              #(self.epoch_num, self.running_cost.average,
              #self.running_acc.accuracy))
        #self.writer.add_summary(
            #create_summary('training average accuracy', self.running_acc.accuracy),
            #self.epoch_num)
        #self.writer.add_summary(
            #create_summary('training average cost', self.running_cost.average),
            #self.epoch_num)

        #self.running_cost.reset()
        #self.running_acc.reset()
