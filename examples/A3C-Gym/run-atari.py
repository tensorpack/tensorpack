#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: run-atari.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import numpy as np
import tensorflow as tf
import os
import sys
import re
import time
import random
import argparse
import six
import cv2

from tensorpack import *
from tensorpack.RL import *
from common import play_one_episode

IMAGE_SIZE = (84, 84)
FRAME_HISTORY = 4
CHANNEL = FRAME_HISTORY * 3
IMAGE_SHAPE3 = IMAGE_SIZE + (CHANNEL,)

NUM_ACTIONS = None
ENV_NAME = None


def get_player(dumpdir=None):
    pl = GymEnv(ENV_NAME, dumpdir=dumpdir, auto_restart=False)
    pl = MapPlayerState(pl, lambda img: cv2.resize(img, IMAGE_SIZE[::-1]))

    global NUM_ACTIONS
    NUM_ACTIONS = pl.get_action_space().num_actions()

    pl = HistoryFramePlayer(pl, FRAME_HISTORY)
    return pl


class Model(ModelDesc):
    def _get_inputs(self):
        assert NUM_ACTIONS is not None
        return [InputDesc(tf.float32, (None,) + IMAGE_SHAPE3, 'state'),
                InputDesc(tf.int32, (None,), 'action'),
                InputDesc(tf.float32, (None,), 'futurereward')]

    def _get_NN_prediction(self, image):
        image = image / 255.0
        with argscope(Conv2D, nl=tf.nn.relu):
            l = Conv2D('conv0', image, out_channel=32, kernel_shape=5)
            l = MaxPooling('pool0', l, 2)
            l = Conv2D('conv1', l, out_channel=32, kernel_shape=5)
            l = MaxPooling('pool1', l, 2)
            l = Conv2D('conv2', l, out_channel=64, kernel_shape=4)
            l = MaxPooling('pool2', l, 2)
            l = Conv2D('conv3', l, out_channel=64, kernel_shape=3)

        l = FullyConnected('fc0', l, 512, nl=tf.identity)
        l = PReLU('prelu', l)
        policy = FullyConnected('fc-pi', l, out_dim=NUM_ACTIONS, nl=tf.identity)
        return policy

    def _build_graph(self, inputs):
        state, action, futurereward = inputs
        policy = self._get_NN_prediction(state)
        self.logits = tf.nn.softmax(policy, name='logits')


def run_submission(cfg, output, nr):
    player = get_player(dumpdir=output)
    predfunc = OfflinePredictor(cfg)
    logger.info("Start evaluation: ")
    for k in range(nr):
        if k != 0:
            player.restart_episode()
        score = play_one_episode(player, predfunc)
        print("Score:", score)


def do_submit(output):
    gym.upload(output, api_key='xxx')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--load', help='load model', required=True)
    parser.add_argument('--env', help='environment name', required=True)
    parser.add_argument('--episode', help='number of episodes to run',
                        type=int, default=100)
    parser.add_argument('--output', help='output directory', default='gym-submit')
    args = parser.parse_args()

    ENV_NAME = args.env
    assert ENV_NAME
    logger.info("Environment Name: {}".format(ENV_NAME))
    p = get_player()
    del p    # set NUM_ACTIONS

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    cfg = PredictConfig(
        model=Model(),
        session_init=SaverRestore(args.load),
        input_names=['state'],
        output_names=['logits'])
    run_submission(cfg, args.output, args.episode)
