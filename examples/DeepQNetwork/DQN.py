#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: DQN.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import numpy as np

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
from tensorpack.RL import *
import tensorflow as tf

from DQNModel import Model as DQNModel
import common
from common import play_model, Evaluator, eval_model_multithread
from atari import AtariPlayer
from expreplay import ExpReplay

BATCH_SIZE = 64
IMAGE_SIZE = (84, 84)
FRAME_HISTORY = 4
ACTION_REPEAT = 4

GAMMA = 0.99

INIT_EXPLORATION = 1
EXPLORATION_EPOCH_ANNEAL = 0.01
END_EXPLORATION = 0.1

MEMORY_SIZE = 1e6
# NOTE: will consume at least 1e6 * 84 * 84 bytes == 6.6G memory.
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
        # create a new axis to stack history on
        pl = MapPlayerState(pl, lambda im: im[:, :, np.newaxis])
        # in training, history is taken care of in expreplay buffer
        pl = HistoryFramePlayer(pl, FRAME_HISTORY)

        pl = PreventStuckPlayer(pl, 30, 1)
    pl = LimitLengthPlayer(pl, 30000)
    return pl


class Model(DQNModel):
    def __init__(self):
        super(Model, self).__init__(IMAGE_SIZE, FRAME_HISTORY, METHOD, NUM_ACTIONS, GAMMA)

    def _get_DQN_prediction(self, image):
        """ image: [0,255]"""
        image = image / 255.0
        with argscope(Conv2D, nl=PReLU.symbolic_function, use_bias=True), \
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
                 # .Conv2D('conv0', out_channel=32, kernel_shape=8, stride=4)
                 # .Conv2D('conv1', out_channel=64, kernel_shape=4, stride=2)
                 # .Conv2D('conv2', out_channel=64, kernel_shape=3)

                 .FullyConnected('fc0', 512, nl=LeakyReLU)())
        if self.method != 'Dueling':
            Q = FullyConnected('fct', l, self.num_actions, nl=tf.identity)
        else:
            # Dueling DQN
            V = FullyConnected('fctV', l, 1, nl=tf.identity)
            As = FullyConnected('fctA', l, self.num_actions, nl=tf.identity)
            Q = tf.add(As, V - tf.reduce_mean(As, 1, keep_dims=True))
        return tf.identity(Q, name='Qvalue')


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

    return TrainConfig(
        dataflow=expreplay,
        callbacks=[
            ModelSaver(),
            ScheduledHyperParamSetter('learning_rate',
                                      [(150, 4e-4), (250, 1e-4), (350, 5e-5)]),
            RunOp(DQNModel.update_target_param),
            expreplay,
            PeriodicTrigger(Evaluator(
                EVAL_EPISODE, ['state'], ['Qvalue'], get_player),
                every_k_epochs=5),
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
            play_model(cfg, get_player(viz=0.01))
        elif args.task == 'eval':
            eval_model_multithread(cfg, EVAL_EPISODE, get_player)
    else:
        config = get_config()
        if args.load:
            config.session_init = SaverRestore(args.load)
        QueueInputTrainer(config).train()
