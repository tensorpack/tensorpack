#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: DQN.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import numpy as np
import tensorflow as tf

import os
import sys
import re
import time
import random
import argparse
import subprocess
import multiprocessing
import threading
from collections import deque

from tensorpack import *
from tensorpack.utils.concurrency import *
from tensorpack.tfutils import symbolic_functions as symbf
from tensorpack.RL import *

import common
from common import play_model, Evaluator, eval_model_multithread
from atari import AtariPlayer
from expreplay import ExpReplay

BATCH_SIZE = 64
IMAGE_SIZE = (84, 84)
FRAME_HISTORY = 4
ACTION_REPEAT = 4

CHANNEL = FRAME_HISTORY
GAMMA = 0.99

INIT_EXPLORATION = 1
EXPLORATION_EPOCH_ANNEAL = 0.01
END_EXPLORATION = 0.1

MEMORY_SIZE = 1e6
# NOTE: will consume at least 1e6 * 84 * 84 bytes == 6.6G memory.
# Suggest using tcmalloc to manage memory space better.
INIT_MEMORY_SIZE = 5e4
STEPS_PER_EPOCH = 10000
EVAL_EPISODE = 50

NUM_ACTIONS = None
ROM_FILE = None
METHOD = None


def get_player(viz=False, train=False):
    pl = AtariPlayer(ROM_FILE, frame_skip=ACTION_REPEAT,
                     image_shape=IMAGE_SIZE[::-1], viz=viz, live_lost_as_eoe=train)
    if not train:
        pl = MapPlayerState(pl, lambda im: im[:, :, np.newaxis])
        pl = HistoryFramePlayer(pl, FRAME_HISTORY)
        pl = PreventStuckPlayer(pl, 30, 1)
    pl = LimitLengthPlayer(pl, 30000)
    return pl


common.get_player = get_player  # so that eval functions in common can use the player


class Model(ModelDesc):
    def _get_inputs(self):
        # use a combined state, where the first channels are the current state,
        # and the last 4 channels are the next state
        return [InputDesc(tf.uint8,
                          (None,) + IMAGE_SIZE + (CHANNEL + 1,),
                          'comb_state'),
                InputDesc(tf.int64, (None,), 'action'),
                InputDesc(tf.float32, (None,), 'reward'),
                InputDesc(tf.bool, (None,), 'isOver')]

    def _get_DQN_prediction(self, image):
        """ image: [0,255]"""
        image = image / 255.0
        with argscope(Conv2D, nl=PReLU.f, use_bias=True), \
                argscope(LeakyReLU, alpha=0.01):
            l = (LinearWrap(image)
                 .Conv2D('conv0', out_channel=32, kernel_shape=5)
                 .MaxPooling('pool0', 2)
                 .Conv2D('conv1', out_channel=32, kernel_shape=5)
                 .MaxPooling('pool1', 2)
                 .Conv2D('conv2', out_channel=64, kernel_shape=4)
                 .MaxPooling('pool2', 2)
                 .Conv2D('conv3', out_channel=64, kernel_shape=3)

                 # the original arch is 2x faster
                 # .Conv2D('conv0', image, out_channel=32, kernel_shape=8, stride=4)
                 # .Conv2D('conv1', out_channel=64, kernel_shape=4, stride=2)
                 # .Conv2D('conv2', out_channel=64, kernel_shape=3)

                 .FullyConnected('fc0', 512, nl=LeakyReLU)())
        if METHOD != 'Dueling':
            Q = FullyConnected('fct', l, NUM_ACTIONS, nl=tf.identity)
        else:
            # Dueling DQN
            V = FullyConnected('fctV', l, 1, nl=tf.identity)
            As = FullyConnected('fctA', l, NUM_ACTIONS, nl=tf.identity)
            Q = tf.add(As, V - tf.reduce_mean(As, 1, keep_dims=True))
        return tf.identity(Q, name='Qvalue')

    def _build_graph(self, inputs):
        comb_state, action, reward, isOver = inputs
        comb_state = tf.cast(comb_state, tf.float32)
        state = tf.slice(comb_state, [0, 0, 0, 0], [-1, -1, -1, 4], name='state')
        self.predict_value = self._get_DQN_prediction(state)
        if not get_current_tower_context().is_training:
            return

        reward = tf.clip_by_value(reward, -1, 1)
        next_state = tf.slice(comb_state, [0, 0, 0, 1], [-1, -1, -1, 4], name='next_state')
        action_onehot = tf.one_hot(action, NUM_ACTIONS, 1.0, 0.0)

        pred_action_value = tf.reduce_sum(self.predict_value * action_onehot, 1)  # N,
        max_pred_reward = tf.reduce_mean(tf.reduce_max(
            self.predict_value, 1), name='predict_reward')
        summary.add_moving_summary(max_pred_reward)

        with tf.variable_scope('target'), \
                collection.freeze_collection([tf.GraphKeys.TRAINABLE_VARIABLES]):
            targetQ_predict_value = self._get_DQN_prediction(next_state)    # NxA

        if METHOD != 'Double':
            # DQN
            best_v = tf.reduce_max(targetQ_predict_value, 1)    # N,
        else:
            # Double-DQN
            sc = tf.get_variable_scope()
            with tf.variable_scope(sc, reuse=True):
                next_predict_value = self._get_DQN_prediction(next_state)
            self.greedy_choice = tf.argmax(next_predict_value, 1)   # N,
            predict_onehot = tf.one_hot(self.greedy_choice, NUM_ACTIONS, 1.0, 0.0)
            best_v = tf.reduce_sum(targetQ_predict_value * predict_onehot, 1)

        target = reward + (1.0 - tf.cast(isOver, tf.float32)) * GAMMA * tf.stop_gradient(best_v)

        self.cost = tf.reduce_mean(symbf.huber_loss(
                                   target - pred_action_value), name='cost')
        summary.add_param_summary(('conv.*/W', ['histogram', 'rms']),
                                  ('fc.*/W', ['histogram', 'rms']))   # monitor all W
        summary.add_moving_summary(self.cost)

    def _get_optimizer(self):
        lr = symbf.get_scalar_var('learning_rate', 1e-3, summary=True)
        opt = tf.train.AdamOptimizer(lr, epsilon=1e-3)
        return optimizer.apply_grad_processors(
            opt, [gradproc.GlobalNormClip(10), gradproc.SummaryGradient()])


def get_config():
    logger.auto_set_dir()

    M = Model()
    expreplay = ExpReplay(
        predictor_io_names=(['state'], ['Qvalue']),
        player=get_player(train=True),
        state_shape=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        memory_size=MEMORY_SIZE,
        init_memory_size=INIT_MEMORY_SIZE,
        exploration=INIT_EXPLORATION,
        end_exploration=END_EXPLORATION,
        exploration_epoch_anneal=EXPLORATION_EPOCH_ANNEAL,
        update_frequency=4,
        history_len=FRAME_HISTORY
    )

    def update_target_param():
        vars = tf.global_variables()
        ops = []
        G = tf.get_default_graph()
        for v in vars:
            target_name = v.op.name
            if target_name.startswith('target'):
                new_name = target_name.replace('target/', '')
                logger.info("{} <- {}".format(target_name, new_name))
                ops.append(v.assign(G.get_tensor_by_name(new_name + ':0')))
        return tf.group(*ops, name='update_target_network')

    return TrainConfig(
        dataflow=expreplay,
        callbacks=[
            ModelSaver(),
            ScheduledHyperParamSetter('learning_rate',
                                      [(150, 4e-4), (250, 1e-4), (350, 5e-5)]),
            RunOp(update_target_param),
            expreplay,
            PeriodicTrigger(Evaluator(
                EVAL_EPISODE, ['state'], ['Qvalue']), every_k_epochs=5),
            # HumanHyperParamSetter('learning_rate', 'hyper.txt'),
            # HumanHyperParamSetter(ObjAttrParam(expreplay, 'exploration'), 'hyper.txt'),
        ],
        model=M,
        steps_per_epoch=STEPS_PER_EPOCH,
        # run the simulator on a separate GPU if available
        predict_tower=[1] if get_nr_gpu() > 1 else [0],
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--load', help='load model')
    parser.add_argument('--task', help='task to perform',
                        choices=['play', 'eval', 'train'], default='train')
    parser.add_argument('--rom', help='atari rom', required=True)
    parser.add_argument('--algo', help='algorithm',
                        choices=['DQN', 'Double', 'Dueling'], default='Double')
    args = parser.parse_args()

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    if args.task != 'train':
        assert args.load is not None
    ROM_FILE = args.rom
    METHOD = args.algo

    # set num_actions
    pl = AtariPlayer(ROM_FILE, viz=False)
    NUM_ACTIONS = pl.get_action_space().num_actions()
    del pl

    if args.task != 'train':
        cfg = PredictConfig(
            model=Model(),
            session_init=get_model_loader(args.load),
            input_names=['state'],
            output_names=['Qvalue'])
        if args.task == 'play':
            play_model(cfg)
        elif args.task == 'eval':
            eval_model_multithread(cfg, EVAL_EPISODE)
    else:
        config = get_config()
        if args.load:
            config.session_init = SaverRestore(args.load)
        QueueInputTrainer(config).train()
