#!/usr/bin/env python2
# -*- coding: utf-8 -*-
# File: DQN.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import numpy as np
import tensorflow as tf

import os, sys, re, time
import random
import argparse
import subprocess
import multiprocessing, threading
from collections import deque

from six.moves import queue
from tqdm import tqdm

from tensorpack import *
from tensorpack.models import  *
from tensorpack.utils import  *
from tensorpack.utils.concurrency import (ensure_proc_terminate, \
        subproc_call, StoppableThread)
from tensorpack.utils.stat import  *
from tensorpack.predict import PredictConfig, get_predict_func, MultiProcessPredictWorker
from tensorpack.tfutils import symbolic_functions as symbf
from tensorpack.callbacks import *

from tensorpack.RL import *

"""
Implement DQN in:
Human-level Control Through Deep Reinforcement Learning
for atari games. Use the variants in:
Deep Reinforcement Learning with Double Q-learning.
"""

BATCH_SIZE = 32
IMAGE_SIZE = (84, 84)
FRAME_HISTORY = 4
ACTION_REPEAT = 3
HEIGHT_RANGE = (36, 204)    # for breakout
CHANNEL = FRAME_HISTORY
IMAGE_SHAPE3 = IMAGE_SIZE + (CHANNEL,)
#HEIGHT_RANGE = (28, -8)   # for pong
GAMMA = 0.99

INIT_EXPLORATION = 1
EXPLORATION_EPOCH_ANNEAL = 0.008
END_EXPLORATION = 0.1

MEMORY_SIZE = 1e6
INIT_MEMORY_SIZE = 50000
STEP_PER_EPOCH = 10000
EVAL_EPISODE = 100

NUM_ACTIONS = None
ROM_FILE = None

def get_player(viz=False, train=False):
    pl = AtariPlayer(ROM_FILE, height_range=HEIGHT_RANGE,
            frame_skip=ACTION_REPEAT, image_shape=IMAGE_SIZE[::-1], viz=viz,
            live_lost_as_eoe=train)
    global NUM_ACTIONS
    NUM_ACTIONS = pl.get_num_actions()
    return pl

class Model(ModelDesc):
    def _get_input_vars(self):
        assert NUM_ACTIONS is not None
        return [InputVar(tf.float32, (None,) + IMAGE_SHAPE3, 'state'),
                InputVar(tf.int64, (None,), 'action'),
                InputVar(tf.float32, (None,), 'reward'),
                InputVar(tf.float32, (None,) + IMAGE_SHAPE3, 'next_state'),
                InputVar(tf.bool, (None,), 'isOver') ]

    def _get_DQN_prediction(self, image, is_training):
        """ image: [0,255]"""
        image = image / 255.0
        with argscope(Conv2D, nl=tf.nn.relu, use_bias=True):
            l = Conv2D('conv0', image, out_channel=32, kernel_shape=5, stride=1)
            l = MaxPooling('pool0', l, 2)
            l = Conv2D('conv1', l, out_channel=32, kernel_shape=5, stride=1)
            l = MaxPooling('pool1', l, 2)
            l = Conv2D('conv2', l, out_channel=64, kernel_shape=4)
            l = MaxPooling('pool2', l, 2)
            l = Conv2D('conv3', l, out_channel=64, kernel_shape=3)

            # the original arch
            #l = Conv2D('conv0', image, out_channel=32, kernel_shape=8, stride=4)
            #l = Conv2D('conv1', l, out_channel=64, kernel_shape=4, stride=2)
            #l = Conv2D('conv2', l, out_channel=64, kernel_shape=3)

        l = FullyConnected('fc0', l, 512, nl=lambda x, name: LeakyReLU.f(x, 0.01, name))
        l = FullyConnected('fct', l, out_dim=NUM_ACTIONS, nl=tf.identity)
        return l

    def _build_graph(self, inputs, is_training):
        state, action, reward, next_state, isOver = inputs
        self.predict_value = self._get_DQN_prediction(state, is_training)
        action_onehot = tf.one_hot(action, NUM_ACTIONS, 1.0, 0.0)
        pred_action_value = tf.reduce_sum(self.predict_value * action_onehot, 1)    #N,
        max_pred_reward = tf.reduce_mean(tf.reduce_max(
            self.predict_value, 1), name='predict_reward')
        tf.add_to_collection(MOVING_SUMMARY_VARS_KEY, max_pred_reward)
        self.greedy_choice = tf.argmax(self.predict_value, 1)   # N,

        with tf.variable_scope('target'):
            targetQ_predict_value = self._get_DQN_prediction(next_state, False)    # NxA

            # DQN
            #best_v = tf.reduce_max(targetQ_predict_value, 1)    # N,

            # Double-DQN
            predict_onehot = tf.one_hot(self.greedy_choice, NUM_ACTIONS, 1.0, 0.0)
            best_v = tf.reduce_sum(targetQ_predict_value * predict_onehot, 1)

            target = reward + (1.0 - tf.cast(isOver, tf.float32)) * GAMMA * tf.stop_gradient(best_v)

        sqrcost = tf.square(target - pred_action_value)
        abscost = tf.abs(target - pred_action_value)    # robust error func
        cost = tf.select(abscost < 1, sqrcost, abscost)
        summary.add_param_summary([('conv.*/W', ['histogram', 'rms']),
                                   ('fc.*/W', ['histogram', 'rms']) ])   # monitor all W
        self.cost = tf.reduce_mean(cost, name='cost')

    def update_target_param(self):
        vars = tf.trainable_variables()
        ops = []
        for v in vars:
            target_name = v.op.name
            if target_name.startswith('target'):
                new_name = target_name.replace('target/', '')
                logger.info("{} <- {}".format(target_name, new_name))
                ops.append(v.assign(tf.get_default_graph().get_tensor_by_name(new_name + ':0')))
        return tf.group(*ops, name='update_target_network')

    def get_gradient_processor(self):
        return [MapGradient(lambda grad: \
                tf.clip_by_global_norm([grad], 5)[0][0]),
                SummaryGradient()]

    def predictor(self, state):
        return self.predict_value.eval(feed_dict={'state:0': [state]})[0]

def play_one_episode(player, func, verbose=False):
    while True:
        s = player.current_state()
        outputs = func([[s]])
        action_value = outputs[0][0]
        act = action_value.argmax()
        if verbose:
            print action_value, act
        if random.random() < 0.01:
            act = random.choice(range(NUM_ACTIONS))
        if verbose:
            print(act)
        reward, isOver = player.action(act)
        if isOver:
            sc = player.stats['score'][0]
            player.reset_stat()
            return sc

def play_model(model_path):
    player = PreventStuckPlayer(HistoryFramePlayer(get_player(0.013), FRAME_HISTORY), 30, 1)
    cfg = PredictConfig(
            model=Model(),
            input_data_mapping=[0],
            session_init=SaverRestore(model_path),
            output_var_names=['fct/output:0'])
    predfunc = get_predict_func(cfg)
    while True:
        score = play_one_episode(player, predfunc)
        print("Total:", score)

def eval_with_funcs(predict_funcs):
    class Worker(StoppableThread):
        def __init__(self, func, queue):
            super(Worker, self).__init__()
            self.func = func
            self.q = queue
        def run(self):
            player = PreventStuckPlayer(HistoryFramePlayer(get_player(), FRAME_HISTORY), 30, 1)
            while not self.stopped():
                score = play_one_episode(player, self.func)
                while not self.stopped():
                    try:
                        self.q.put(score, timeout=5)
                        break
                    except queue.Queue.Full:
                        pass

    q = queue.Queue()
    threads = [Worker(f, q) for f in predict_funcs]

    for k in threads:
        k.start()
        time.sleep(0.1) # avoid simulator bugs
    stat = StatCounter()
    try:
        for _ in tqdm(range(EVAL_EPISODE)):
            r = q.get()
            stat.feed(r)
        for k in threads: k.stop()
        for k in threads: k.join()
    finally:
        return (stat.average, stat.max)

def eval_model_multithread(model_path):
    cfg = PredictConfig(
            model=Model(),
            input_data_mapping=[0],
            session_init=SaverRestore(model_path),
            output_var_names=['fct/output:0'])
    p = get_player(); del p # set NUM_ACTIONS
    func = get_predict_func(cfg)
    NR_PROC = min(multiprocessing.cpu_count() // 2, 8)
    mean, max = eval_with_funcs([func] * NR_PROC)
    logger.info("Average Score: {}; Max Score: {}".format(mean, max))

class Evaluator(Callback):
    def _before_train(self):
        NR_PROC = min(multiprocessing.cpu_count() // 2, 8)
        self.pred_funcs = [self.trainer.get_predict_func(
           ['state'], ['fct/output'])] * NR_PROC

    def _trigger_epoch(self):
        mean, max = eval_with_funcs(self.pred_funcs)
        self.trainer.write_scalar_summary('mean_score', mean)
        self.trainer.write_scalar_summary('max_score', max)

def get_config():
    basename = os.path.basename(__file__)
    logger.set_logger_dir(
        os.path.join('train_log', basename[:basename.rfind('.')]))

    M = Model()
    dataset_train = ExpReplay(
            predictor=M.predictor,
            player=get_player(train=True),
            num_actions=NUM_ACTIONS,
            memory_size=MEMORY_SIZE,
            batch_size=BATCH_SIZE,
            populate_size=INIT_MEMORY_SIZE,
            exploration=INIT_EXPLORATION,
            end_exploration=END_EXPLORATION,
            exploration_epoch_anneal=EXPLORATION_EPOCH_ANNEAL,
            update_frequency=4,
            reward_clip=(-1, 1),
            history_len=FRAME_HISTORY)

    lr = tf.Variable(0.00025, trainable=False, name='learning_rate')
    tf.scalar_summary('learning_rate', lr)

    return TrainConfig(
        dataset=dataset_train,
        optimizer=tf.train.AdamOptimizer(lr, epsilon=1e-3),
        callbacks=Callbacks([
            StatPrinter(),
            ModelSaver(),
            HumanHyperParamSetter('learning_rate', 'hyper.txt'),
            HumanHyperParamSetter(ObjAttrParam(dataset_train, 'exploration'), 'hyper.txt'),
            RunOp(lambda: M.update_target_param()),
            dataset_train,
            PeriodicCallback(Evaluator(), 2),
        ]),
        # save memory for multiprocess evaluator
        session_config=get_default_sess_config(0.3),
        model=M,
        step_per_epoch=STEP_PER_EPOCH,
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.') # nargs='*' in multi mode
    parser.add_argument('--load', help='load model')
    parser.add_argument('--task', help='task to perform',
            choices=['play', 'eval', 'train'], default='train')
    parser.add_argument('--rom', help='atari rom', required=True)
    args = parser.parse_args()

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if args.task != 'train':
        assert args.load is not None

    ROM_FILE = args.rom

    if args.task == 'play':
        play_model(args.load)
        sys.exit()
    if args.task == 'eval':
        eval_model_multithread(args.load)
        sys.exit()

    with tf.Graph().as_default():
        config = get_config()
        if args.load:
            config.session_init = SaverRestore(args.load)
        SimpleTrainer(config).train()

