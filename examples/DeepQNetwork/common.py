#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: common.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>
import random
import time
import threading
import multiprocessing
import numpy as np
import cv2
from collections import deque
from tqdm import tqdm
from six.moves import queue

import gym
from gym import spaces

from tensorpack.utils.concurrency import StoppableThread, ShareSessionThread
from tensorpack.callbacks import Triggerable
from tensorpack.utils import logger
from tensorpack.utils.stats import StatCounter
from tensorpack.utils.utils import get_tqdm_kwargs


def play_one_episode(env, func, render=False):
    def predict(s):
        """
        Map from observation to action, with 0.001 greedy.
        """
        act = func([[s]])[0][0].argmax()
        if random.random() < 0.001:
            spc = env.action_space
            act = spc.sample()
        return act

    ob = env.reset()
    sum_r = 0
    while True:
        act = predict(ob)
        ob, r, isOver, info = env.step(act)
        if render:
            env.render()
        sum_r += r
        if isOver:
            return sum_r


def play_n_episodes(player, predfunc, nr, render=False):
    logger.info("Start Playing ... ")
    for k in range(nr):
        score = play_one_episode(player, predfunc, render=render)
        print("{}/{}, score={}".format(k, nr, score))


def eval_with_funcs(predictors, nr_eval, get_player_fn):
    """
    Args:
        predictors ([PredictorBase])
    """
    class Worker(StoppableThread, ShareSessionThread):
        def __init__(self, func, queue):
            super(Worker, self).__init__()
            self._func = func
            self.q = queue

        def func(self, *args, **kwargs):
            if self.stopped():
                raise RuntimeError("stopped!")
            return self._func(*args, **kwargs)

        def run(self):
            with self.default_sess():
                player = get_player_fn(train=False)
                while not self.stopped():
                    try:
                        score = play_one_episode(player, self.func)
                        # print("Score, ", score)
                    except RuntimeError:
                        return
                    self.queue_put_stoppable(self.q, score)

    q = queue.Queue()
    threads = [Worker(f, q) for f in predictors]

    for k in threads:
        k.start()
        time.sleep(0.1)  # avoid simulator bugs
    stat = StatCounter()
    try:
        for _ in tqdm(range(nr_eval), **get_tqdm_kwargs()):
            r = q.get()
            stat.feed(r)
        logger.info("Waiting for all the workers to finish the last run...")
        for k in threads:
            k.stop()
        for k in threads:
            k.join()
        while q.qsize():
            r = q.get()
            stat.feed(r)
    except:
        logger.exception("Eval")
    finally:
        if stat.count > 0:
            return (stat.average, stat.max)
        return (0, 0)


def eval_model_multithread(pred, nr_eval, get_player_fn):
    """
    Args:
        pred (OfflinePredictor): state -> Qvalue
    """
    NR_PROC = min(multiprocessing.cpu_count() // 2, 8)
    with pred.sess.as_default():
        mean, max = eval_with_funcs([pred] * NR_PROC, nr_eval, get_player_fn)
    logger.info("Average Score: {}; Max Score: {}".format(mean, max))


class Evaluator(Triggerable):
    def __init__(self, nr_eval, input_names, output_names, get_player_fn):
        self.eval_episode = nr_eval
        self.input_names = input_names
        self.output_names = output_names
        self.get_player_fn = get_player_fn

    def _setup_graph(self):
        NR_PROC = min(multiprocessing.cpu_count() // 2, 20)
        self.pred_funcs = [self.trainer.get_predictor(
            self.input_names, self.output_names)] * NR_PROC

    def _trigger(self):
        t = time.time()
        mean, max = eval_with_funcs(
            self.pred_funcs, self.eval_episode, self.get_player_fn)
        t = time.time() - t
        if t > 10 * 60:  # eval takes too long
            self.eval_episode = int(self.eval_episode * 0.94)
        self.trainer.monitors.put_scalar('mean_score', mean)
        self.trainer.monitors.put_scalar('max_score', max)


"""
------------------------------------------------------------------------------
The following wrappers are copied or modified from openai/baselines:
https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py
"""


class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env, shape):
        gym.ObservationWrapper.__init__(self, env)
        self.shape = shape
        obs = env.observation_space
        assert isinstance(obs, spaces.Box)
        chan = 1 if len(obs.shape) == 2 else obs.shape[2]
        shape3d = shape if chan == 1 else shape + (chan,)
        self.observation_space = spaces.Box(low=0, high=255, shape=shape3d)

    def _observation(self, obs):
        return cv2.resize(obs, self.shape)


class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        """Buffer observations and stack across channels (last axis)."""
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        chan = 1 if len(shp) == 2 else shp[2]
        self._base_dim = len(shp)
        self.observation_space = spaces.Box(low=0, high=255, shape=(shp[0], shp[1], chan * k))

    def _reset(self):
        """Clear buffer and re-fill by duplicating the first observation."""
        ob = self.env.reset()
        for _ in range(self.k - 1):
            self.frames.append(np.zeros_like(ob))
        self.frames.append(ob)
        return self._observation()

    def _step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        return self._observation(), reward, done, info

    def _observation(self):
        assert len(self.frames) == self.k
        if self._base_dim == 2:
            return np.stack(self.frames, axis=-1)
        else:
            return np.concatenate(self.frames, axis=2)


class _FireResetEnv(gym.Wrapper):
    def __init__(self, env):
        """Take action on reset for environments that are fixed until firing."""
        gym.Wrapper.__init__(self, env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def _reset(self):
        self.env.reset()
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset()
        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset()
        return obs


def FireResetEnv(env):
    if isinstance(env, gym.Wrapper):
        baseenv = env.unwrapped
    else:
        baseenv = env
    if 'FIRE' in baseenv.get_action_meanings():
        return _FireResetEnv(env)
    return env


class LimitLength(gym.Wrapper):
    def __init__(self, env, k):
        gym.Wrapper.__init__(self, env)
        self.k = k

    def _reset(self):
        # This assumes that reset() will really reset the env.
        # If the underlying env tries to be smart about reset
        # (e.g. end-of-life), the assumption doesn't hold.
        ob = self.env.reset()
        self.cnt = 0
        return ob

    def _step(self, action):
        ob, r, done, info = self.env.step(action)
        self.cnt += 1
        if self.cnt == self.k:
            done = True
        return ob, r, done, info
