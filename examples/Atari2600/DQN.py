#!/usr/bin/env python2
# -*- coding: utf-8 -*-
# File: DQN.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import numpy as np
import tensorflow as tf
import os, sys, re
import random
import argparse
from tqdm import tqdm
import subprocess
import multiprocessing
from collections import deque

from tensorpack import *
from tensorpack.models import  *
from tensorpack.utils import  *
from tensorpack.utils.concurrency import ensure_proc_terminate, subproc_call
from tensorpack.utils.stat import  *
from tensorpack.predict import PredictConfig, get_predict_func, ParallelPredictWorker
from tensorpack.tfutils import symbolic_functions as symbf
from tensorpack.callbacks import *

from tensorpack.dataflow.dataset import AtariPlayer
from tensorpack.dataflow.RL import ExpReplay

"""
Implement DQN in:
Human-level control through deep reinforcement learning
for atari games
"""

BATCH_SIZE = 32
IMAGE_SIZE = 84
FRAME_HISTORY = 4
ACTION_REPEAT = 4
HEIGHT_RANGE = (36, 204)    # for breakout
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

def get_player(viz=False):
    pl = AtariPlayer(ROM_FILE, viz=viz, height_range=HEIGHT_RANGE, frame_skip=ACTION_REPEAT)
    global NUM_ACTIONS
    NUM_ACTIONS = pl.get_num_actions()
    return pl

class Model(ModelDesc):
    def _get_input_vars(self):
        assert NUM_ACTIONS is not None
        return [InputVar(tf.float32, (None, IMAGE_SIZE, IMAGE_SIZE, FRAME_HISTORY), 'state'),
                InputVar(tf.int32, (None,), 'action'),
                InputVar(tf.float32, (None,), 'reward'),
                InputVar(tf.float32, (None, IMAGE_SIZE, IMAGE_SIZE, FRAME_HISTORY), 'next_state'),
                InputVar(tf.bool, (None,), 'isOver') ]

    def _get_DQN_prediction(self, image, is_training):
        """ image: [0,255]"""
        image = image / 255.0
        with argscope(Conv2D, nl=PReLU.f, use_bias=True):
            l = Conv2D('conv0', image, out_channel=32, kernel_shape=5, stride=1)
            l = MaxPooling('pool0', l, 2)
            l = Conv2D('conv1', l, out_channel=32, kernel_shape=5, stride=1)
            l = MaxPooling('pool1', l, 2)
            l = Conv2D('conv2', l, out_channel=64, kernel_shape=4)
            l = MaxPooling('pool2', l, 2)
            l = Conv2D('conv3', l, out_channel=64, kernel_shape=3)
            #l = MaxPooling('pool3', l, 2)
            #l = Conv2D('conv4', l, out_channel=64, kernel_shape=3)

        l = FullyConnected('fc0', l, 512, nl=lambda x, name: LeakyReLU.f(x, 0.01, name))
        l = FullyConnected('fct', l, out_dim=NUM_ACTIONS, nl=tf.identity)
        return l

    def _build_graph(self, inputs, is_training):
        state, action, reward, next_state, isOver = inputs
        self.predict_value = self._get_DQN_prediction(state, is_training)
        action_onehot = symbf.one_hot(action, NUM_ACTIONS)
        pred_action_value = tf.reduce_sum(self.predict_value * action_onehot, 1)    #Nx1
        max_pred_reward = tf.reduce_mean(tf.reduce_max(
            self.predict_value, 1), name='predict_reward')
        tf.add_to_collection(MOVING_SUMMARY_VARS_KEY, max_pred_reward)

        with tf.variable_scope('target'):
            targetQ_predict_value = tf.stop_gradient(
                    self._get_DQN_prediction(next_state, False))    # NxA
            target = reward + (1.0 - tf.cast(isOver, tf.float32)) * \
                    GAMMA * tf.reduce_max(targetQ_predict_value, 1)    # Nx1

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

def current_predictor(state):
    pred_var = tf.get_default_graph().get_tensor_by_name('fct/output:0')
    pred = pred_var.eval(feed_dict={'state:0': [state]})
    return pred[0]

def play_one_episode(player, func, verbose=False):
    tot_reward = 0
    que = deque(maxlen=30)
    while True:
        s = player.current_state()
        outputs = func([[s]])
        action_value = outputs[0][0]
        act = action_value.argmax()
        if verbose:
            print action_value, act
        if random.random() < 0.01:
            act = random.choice(range(NUM_ACTIONS))
        if len(que) == que.maxlen \
                and que.count(que[0]) == que.maxlen:
            act = 1 # hack, avoid stuck
        que.append(act)
        if verbose:
            print(act)
        reward, isOver = player.action(act)
        tot_reward += reward
        if isOver:
            return tot_reward

def play_model(model_path):
    player = HistoryFramePlayer(get_player(0.01), FRAME_HISTORY)
    cfg = PredictConfig(
            model=Model(),
            input_data_mapping=[0],
            session_init=SaverRestore(model_path),
            output_var_names=['fct/output:0'])
    predfunc = get_predict_func(cfg)
    while True:
        score = play_one_episode(player, predfunc)
        print("Total:", score)

def eval_model_multiprocess(model_path):
    M = Model()
    cfg = PredictConfig(
            model=M,
            input_data_mapping=[0],
            session_init=SaverRestore(model_path),
            output_var_names=['fct/output:0'])

    class Worker(MultiProcessPredictWorker):
        def __init__(self, idx, gpuid, config, outqueue):
            super(Worker, self).__init__(idx, gpuid, config)
            self.outq = outqueue

        def run(self):
            player = HistoryFramePlayer(get_player(), FRAME_HISTORY)
            self._init_runtime()
            while True:
                score = play_one_episode(player, self.func)
                self.outq.put(score)

    NR_PROC = min(multiprocessing.cpu_count() // 2, 10)
    procs = []
    q = multiprocessing.Queue()
    for k in range(NR_PROC):
        procs.append(Worker(k, -1, cfg, q))
    ensure_proc_terminate(procs)
    for k in procs:
        k.start()
    stat = StatCounter()
    try:
        for _ in tqdm(range(EVAL_EPISODE)):
            r = q.get()
            stat.feed(r)
    finally:
        logger.info("Average Score: {}; Max Score: {}".format(
            stat.average, stat.max))

class Evaluator(Callback):
    def _trigger_epoch(self):
        logger.info("Evaluating...")
        output = subproc_call(
                "CUDA_VISIBLE_DEVICES=  {} --task eval --rom {} --load {}".format(
                sys.argv[0], romfile, os.path.join(logger.LOG_DIR, 'checkpoint')),
                timeout=10*60)
        if output:
            last = output.strip().split('\n')[-1]
            last = last[last.find(']')+1:]
            mean, maximum = re.findall('[0-9\.\-]+', last)[-2:]
            self.trainer.write_scalar_summary('mean_score', mean)
            self.trainer.write_scalar_summary('max_score', maximum)

def get_config():
    basename = os.path.basename(__file__)
    logger.set_logger_dir(
        os.path.join('train_log', basename[:basename.rfind('.')]))

    M = Model()
    dataset_train = ExpReplay(
            predictor=current_predictor,
            player=get_player(),
            num_actions=NUM_ACTIONS,
            memory_size=MEMORY_SIZE,
            batch_size=BATCH_SIZE,
            populate_size=INIT_MEMORY_SIZE,
            exploration=INIT_EXPLORATION,
            end_exploration=END_EXPLORATION,
            exploration_epoch_anneal=EXPLORATION_EPOCH_ANNEAL,
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
        eval_model_multiprocess(args.load)
        sys.exit()

    with tf.Graph().as_default():
        config = get_config()
        if args.load:
            config.session_init = SaverRestore(args.load)
        SimpleTrainer(config).train()

