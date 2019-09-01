#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: DQN.py
# Author: Yuxin Wu

import argparse
import numpy as np
import os
import cv2
import gym
import tensorflow as tf

from tensorpack import *

from atari_wrapper import FireResetEnv, FrameStack, LimitLength, MapState
from common import Evaluator, eval_model_multithread, play_n_episodes
from DQNModel import Model as DQNModel
from expreplay import ExpReplay

BATCH_SIZE = 64
IMAGE_SIZE = (84, 84)
FRAME_HISTORY = 4
UPDATE_FREQ = 4  # the number of new state transitions per parameter update (per training step)

MEMORY_SIZE = 1e6
# will consume at least 1e6 * 84 * 84 bytes == 6.6G memory.
INIT_MEMORY_SIZE = MEMORY_SIZE // 20
STEPS_PER_EPOCH = 100000 // UPDATE_FREQ  # each epoch is 100k state transitions
NUM_PARALLEL_PLAYERS = 3

USE_GYM = False
ENV_NAME = None


def resize_keepdims(im, size):
    # Opencv's resize remove the extra dimension for grayscale images. We add it back.
    ret = cv2.resize(im, size)
    if im.ndim == 3 and ret.ndim == 2:
        ret = ret[:, :, np.newaxis]
    return ret


def get_player(viz=False, train=False):
    if USE_GYM:
        env = gym.make(ENV_NAME)
    else:
        from atari import AtariPlayer
        env = AtariPlayer(ENV_NAME, frame_skip=4, viz=viz,
                          live_lost_as_eoe=train, max_num_frames=60000)
    env = FireResetEnv(env)
    env = MapState(env, lambda im: resize_keepdims(im, IMAGE_SIZE))
    if not train:
        # in training, history is taken care of in expreplay buffer
        env = FrameStack(env, FRAME_HISTORY)
    if train and USE_GYM:
        env = LimitLength(env, 60000)
    return env


class Model(DQNModel):
    """
    A DQN model for 2D/3D (image) observations.
    """
    def _get_DQN_prediction(self, image):
        assert image.shape.rank in [4, 5], image.shape
        # image: N, H, W, (C), Hist
        if image.shape.rank == 5:
            # merge C & Hist
            image = tf.reshape(
                image,
                [-1] + list(self.state_shape[:2]) + [self.state_shape[2] * FRAME_HISTORY])

        image = image / 255.0
        with argscope(Conv2D, activation=lambda x: PReLU('prelu', x), use_bias=True):
            l = (LinearWrap(image)
                 # Nature architecture
                 .Conv2D('conv0', 32, 8, strides=4)
                 .Conv2D('conv1', 64, 4, strides=2)
                 .Conv2D('conv2', 64, 3)

                 # architecture used for the figure in the README, slower but takes fewer iterations to converge
                 # .Conv2D('conv0', out_channel=32, kernel_shape=5)
                 # .MaxPooling('pool0', 2)
                 # .Conv2D('conv1', out_channel=32, kernel_shape=5)
                 # .MaxPooling('pool1', 2)
                 # .Conv2D('conv2', out_channel=64, kernel_shape=4)
                 # .MaxPooling('pool2', 2)
                 # .Conv2D('conv3', out_channel=64, kernel_shape=3)

                 .FullyConnected('fc0', 512)
                 .tf.nn.leaky_relu(alpha=0.01)())
        if self.method != 'Dueling':
            Q = FullyConnected('fct', l, self.num_actions)
        else:
            # Dueling DQN
            V = FullyConnected('fctV', l, 1)
            As = FullyConnected('fctA', l, self.num_actions)
            Q = tf.add(As, V - tf.reduce_mean(As, 1, keep_dims=True))
        return tf.identity(Q, name='Qvalue')


def get_config(model):
    global args
    expreplay = ExpReplay(
        predictor_io_names=(['state'], ['Qvalue']),
        get_player=lambda: get_player(train=True),
        num_parallel_players=NUM_PARALLEL_PLAYERS,
        state_shape=model.state_shape,
        batch_size=BATCH_SIZE,
        memory_size=MEMORY_SIZE,
        init_memory_size=INIT_MEMORY_SIZE,
        update_frequency=UPDATE_FREQ,
        history_len=FRAME_HISTORY,
        state_dtype=model.state_dtype.as_numpy_dtype
    )

    # Set to other values if you need a different initial exploration
    # (e.g., # if you're resuming a training half-way)
    # expreplay.exploration = 1.0

    return TrainConfig(
        data=QueueInput(expreplay),
        model=model,
        callbacks=[
            ModelSaver(),
            PeriodicTrigger(
                RunOp(DQNModel.update_target_param, verbose=True),
                every_k_steps=5000),    # update target network every 5k steps
            expreplay,
            ScheduledHyperParamSetter('learning_rate',
                                      [(0, 1e-3), (60, 5e-4), (400, 1e-4)]),
            ScheduledHyperParamSetter(
                ObjAttrParam(expreplay, 'exploration'),
                [(0, 1), (10, 0.1), (400, 0.01)],   # 1->0.1 in the first million steps
                interp='linear'),
            PeriodicTrigger(Evaluator(
                args.num_eval, ['state'], ['Qvalue'], get_player),
                every_k_epochs=5 if 'pong' in args.env.lower() else 10),  # eval more frequently for easy games
        ],
        steps_per_epoch=STEPS_PER_EPOCH,
        max_epoch=500,  # a total of 50M state transition
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--load', help='load model')
    parser.add_argument('--task', help='task to perform',
                        choices=['play', 'eval', 'train'], default='train')
    parser.add_argument('--env', required=True,
                        help='either an atari rom file (that ends with .bin) or a gym atari environment name')
    parser.add_argument('--algo', help='algorithm',
                        choices=['DQN', 'Double', 'Dueling'], default='Double')
    parser.add_argument('--num-eval', default=50, type=int)
    args = parser.parse_args()
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    ENV_NAME = args.env
    USE_GYM = not ENV_NAME.endswith('.bin')

    # set num_actions
    num_actions = get_player().action_space.n
    logger.info("ENV: {}, Num Actions: {}".format(args.env, num_actions))

    state_shape = IMAGE_SIZE + (3, ) if USE_GYM else IMAGE_SIZE
    model = Model(state_shape, FRAME_HISTORY, args.algo, num_actions)

    if args.task != 'train':
        assert args.load is not None
        pred = OfflinePredictor(PredictConfig(
            model=model,
            session_init=SmartInit(args.load),
            input_names=['state'],
            output_names=['Qvalue']))
        if args.task == 'play':
            play_n_episodes(get_player(viz=0.01), pred, 100, render=True)
        elif args.task == 'eval':
            eval_model_multithread(pred, args.num_eval, get_player)
    else:
        logger.set_logger_dir(
            os.path.join('train_log', 'DQN-{}'.format(
                os.path.basename(args.env).split('.')[0])))
        config = get_config(model)
        config.session_init = SmartInit(args.load)
        launch_train_with_config(config, SimpleTrainer())
