#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: DQN.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import os
import argparse
import cv2
import tensorflow as tf


from tensorpack import *

from DQNModel import Model as DQNModel
from common import Evaluator, eval_model_multithread, play_n_episodes
from atari_wrapper import FrameStack, MapState, FireResetEnv
from expreplay import ExpReplay
from atari import AtariPlayer

BATCH_SIZE = 64
IMAGE_SIZE = (84, 84)
FRAME_HISTORY = 4
ACTION_REPEAT = 4   # aka FRAME_SKIP
UPDATE_FREQ = 4

GAMMA = 0.99

MEMORY_SIZE = 1e6
# will consume at least 1e6 * 84 * 84 bytes == 6.6G memory.
INIT_MEMORY_SIZE = MEMORY_SIZE // 20
STEPS_PER_EPOCH = 10000 // UPDATE_FREQ * 10  # each epoch is 100k played frames
EVAL_EPISODE = 50

NUM_ACTIONS = None
ROM_FILE = None
METHOD = None


def get_player(viz=False, train=False):
    env = AtariPlayer(ROM_FILE, frame_skip=ACTION_REPEAT, viz=viz,
                      live_lost_as_eoe=train, max_num_frames=60000)
    env = FireResetEnv(env)
    env = MapState(env, lambda im: cv2.resize(im, IMAGE_SIZE))
    if not train:
        # in training, history is taken care of in expreplay buffer
        env = FrameStack(env, FRAME_HISTORY)
    return env


class Model(DQNModel):
    def __init__(self):
        super(Model, self).__init__(IMAGE_SIZE, FRAME_HISTORY, METHOD, NUM_ACTIONS, GAMMA)

    def _get_DQN_prediction(self, image):
        """ image: [0,255]"""
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


def get_config():
    expreplay = ExpReplay(
        predictor_io_names=(['state'], ['Qvalue']),
        player=get_player(train=True),
        state_shape=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        memory_size=MEMORY_SIZE,
        init_memory_size=INIT_MEMORY_SIZE,
        init_exploration=1.0,
        update_frequency=UPDATE_FREQ,
        history_len=FRAME_HISTORY
    )

    return TrainConfig(
        data=QueueInput(expreplay),
        model=Model(),
        callbacks=[
            ModelSaver(),
            PeriodicTrigger(
                RunOp(DQNModel.update_target_param, verbose=True),
                every_k_steps=10000 // UPDATE_FREQ),    # update target network every 10k steps
            expreplay,
            ScheduledHyperParamSetter('learning_rate',
                                      [(60, 4e-4), (100, 2e-4)]),
            ScheduledHyperParamSetter(
                ObjAttrParam(expreplay, 'exploration'),
                [(0, 1), (10, 0.1), (320, 0.01)],   # 1->0.1 in the first million steps
                interp='linear'),
            PeriodicTrigger(Evaluator(
                EVAL_EPISODE, ['state'], ['Qvalue'], get_player),
                every_k_epochs=10),
            HumanHyperParamSetter('learning_rate'),
        ],
        steps_per_epoch=STEPS_PER_EPOCH,
        max_epoch=1000,
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
    ROM_FILE = args.rom
    METHOD = args.algo
    # set num_actions
    NUM_ACTIONS = AtariPlayer(ROM_FILE).action_space.n
    logger.info("ROM: {}, Num Actions: {}".format(ROM_FILE, NUM_ACTIONS))

    if args.task != 'train':
        assert args.load is not None
        pred = OfflinePredictor(PredictConfig(
            model=Model(),
            session_init=get_model_loader(args.load),
            input_names=['state'],
            output_names=['Qvalue']))
        if args.task == 'play':
            play_n_episodes(get_player(viz=0.01), pred, 100)
        elif args.task == 'eval':
            eval_model_multithread(pred, EVAL_EPISODE, get_player)
    else:
        logger.set_logger_dir(
            os.path.join('train_log', 'DQN-{}'.format(
                os.path.basename(ROM_FILE).split('.')[0])))
        config = get_config()
        if args.load:
            config.session_init = get_model_loader(args.load)
        launch_train_with_config(config, SimpleTrainer())
