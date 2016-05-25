#!/usr/bin/env python2
# -*- coding: utf-8 -*-
# File: RL.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

from abc import abstractmethod, ABCMeta
import random
import numpy as np
from collections import deque, namedtuple, defaultdict
from tqdm import tqdm
import cv2
import six

from .base import DataFlow
from tensorpack.utils import *
from tensorpack.callbacks.base import Callback

"""
Implement RL-related data preprocessing
"""

__all__ = ['ExpReplay', 'RLEnvironment', 'NaiveRLEnvironment']

Experience = namedtuple('Experience',
        ['state', 'action', 'reward', 'next', 'isOver'])

class RLEnvironment(object):
    __meta__ = ABCMeta

    def __init__(self):
        self.reset_stat()

    @abstractmethod
    def current_state(self):
        """
        Observe, return a state representation
        """

    @abstractmethod
    def action(self, act):
        """
        Perform an action
        :params act: the action
        :returns: (reward, isOver)
        """

    @abstractmethod
    def get_stat(self):
        """
        return a dict of statistics (e.g., score) after running for a while
        """

    def reset_stat(self):
        """ reset the statistics counter"""
        self.stats = defaultdict(list)

class NaiveRLEnvironment(RLEnvironment):
    """ for testing only"""
    def __init__(self):
        self.k = 0
    def current_state(self):
        self.k += 1
        return self.k
    def action(self, act):
        self.k = act
        return (self.k, self.k > 10)


class ExpReplay(DataFlow, Callback):
    """
    Implement experience replay in the paper
    `Human-level control through deep reinforcement learning`.
    """
    def __init__(self,
            predictor,
            player,
            num_actions,
            memory_size=1e6,
            batch_size=32,
            populate_size=50000,
            exploration=1,
            end_exploration=0.1,
            exploration_epoch_anneal=0.002,
            reward_clip=None,
            new_experience_per_step=1
            ):
        """
        :param predictor: callabale. called with a state, return a distribution
        :param player: a `RLEnvironment`
        """
        for k, v in locals().items():
            if k != 'self':
                setattr(self, k, v)
        logger.info("Number of Legal actions: {}".format(self.num_actions))
        self.mem = deque(maxlen=memory_size)
        self.rng = get_rng(self)

    def init_memory(self):
        logger.info("Populating replay memory...")
        with tqdm(total=self.populate_size) as pbar:
            while len(self.mem) < self.populate_size:
                self._populate_exp()
                pbar.update()

    def reset_state(self):
        raise RuntimeError("Don't run me in multiple processes")

    def _populate_exp(self):
        p = self.rng.rand()
        old_s = self.player.current_state()
        if p <= self.exploration:
            act = self.rng.choice(range(self.num_actions))
        else:
            act = np.argmax(self.predictor(old_s))  # TODO race condition in session?
        reward, isOver = self.player.action(act)
        if self.reward_clip:
            reward = np.clip(reward, self.reward_clip[0], self.reward_clip[1])
        s = self.player.current_state()

        #def view_state(state):
            #""" for debug state representation"""
            #r = np.concatenate([state[:,:,k] for k in range(state.shape[2])], axis=1)
            #print r.shape
            #cv2.imshow("state", r)
            #cv2.waitKey()
        #print act, reward
        #view_state(s)

        # s is considered useless if isOver==True
        self.mem.append(Experience(old_s, act, reward, s, isOver))

    def get_data(self):
        while True:
            idxs = self.rng.randint(len(self.mem), size=self.batch_size)
            batch_exp = [self.mem[k] for k in idxs]
            yield self._process_batch(batch_exp)
            for _ in range(self.new_experience_per_step):
                self._populate_exp()

    def _process_batch(self, batch_exp):
        state_shape = batch_exp[0].state.shape
        state = np.zeros((self.batch_size, ) + state_shape, dtype='float32')
        next_state = np.zeros((self.batch_size, ) + state_shape, dtype='float32')
        reward = np.zeros((self.batch_size,), dtype='float32')
        action = np.zeros((self.batch_size,), dtype='int32')
        isOver = np.zeros((self.batch_size,), dtype='bool')

        for idx, b in enumerate(batch_exp):
            state[idx] = b.state
            action[idx] = b.action
            next_state[idx] = b.next
            reward[idx] = b.reward
            isOver[idx] = b.isOver
        return [state, action, reward, next_state, isOver]

    # Callback-related:

    def _before_train(self):
        self.init_memory()

    def _trigger_epoch(self):
        if self.exploration > self.end_exploration:
            self.exploration -= self.exploration_epoch_anneal
            logger.info("Exploration changed to {}".format(self.exploration))
        stats = self.player.get_stat()
        for k, v in six.iteritems(stats):
            if isinstance(v, float):
                self.trainer.write_scalar_summary('expreplay/' + k, v)
        self.player.reset_stat()


if __name__ == '__main__':
    from tensorpack.dataflow.dataset import AtariDriver, AtariPlayer
    predictor = lambda x: np.array([1,1,1,1])
    predictor.initialized = False
    E = AtariExpReplay(predictor, predictor,
            AtariPlayer(AtariDriver('../../space_invaders.bin', viz=0.01)),
            populate_size=1000)
    E.init_memory()

    for k in E.get_data():
        pass
        #import IPython;
        #IPython.embed(config=IPython.terminal.ipapp.load_default_config())
        #break
