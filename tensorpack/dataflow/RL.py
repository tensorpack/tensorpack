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

__all__ = ['ExpReplay', 'RLEnvironment', 'NaiveRLEnvironment', 'HistoryFramePlayer']

Experience = namedtuple('Experience',
        ['state', 'action', 'reward', 'isOver'])

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

class ProxyPlayer(RLEnvironment):
    def __init__(self, player):
        self.player = player

    def get_stat(self):
        return self.player.get_stat()

    def reset_stat(self):
        self.player.reset_stat()

    def current_state(self):
        return self.player.current_state()

    def action(self, act):
        return self.player.action(act)

class HistoryFramePlayer(ProxyPlayer):
    def __init__(self, player, hist_len):
        super(HistoryFramePlayer, self).__init__(player)
        self.history = deque(maxlen=hist_len)

        s = self.player.current_state()
        self.history.append(s)

    def current_state(self):
        assert len(self.history) != 0
        diff_len = self.history.maxlen - len(self.history)
        if diff_len == 0:
            return np.concatenate(self.history, axis=2)
        zeros = [np.zeros_like(self.history[0]) for k in range(diff_len)]
        for k in self.history:
            zeros.append(k)
        return np.concatenate(zeros, axis=2)

    def action(self, act):
        r, isOver = self.player.action(act)
        s = self.player.current_state()
        self.history.append(s)

        if isOver:  # s would be a new episode
            self.history.clear()
            self.history.append(s)
        return (r, isOver)

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
            new_experience_per_step=1,
            history_len=1
            ):
        """
        :param predictor: a callabale calling the up-to-date network.
            called with a state, return a distribution
        :param player: a `RLEnvironment`
        :param num_actions: int
        :param history_len: length of history frames to concat. zero-filled initial frames
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
        old_s = self.player.current_state()
        if self.rng.rand() <= self.exploration:
            act = self.rng.choice(range(self.num_actions))
        else:
            # build a history state
            ss = [old_s]

            isOver = False
            for k in range(1, self.history_len):
                hist_exp = self.mem[-k]
                if hist_exp.isOver:
                    isOver = True
                if isOver:
                    ss.append(np.zeros_like(ss[0]))
                else:
                    ss.append(hist_exp.state)
            ss.reverse()
            ss = np.concatenate(ss, axis=2)
            act = np.argmax(self.predictor(ss))
        reward, isOver = self.player.action(act)
        if self.reward_clip:
            reward = np.clip(reward, self.reward_clip[0], self.reward_clip[1])
        self.mem.append(Experience(old_s, act, reward, isOver))

    def get_data(self):
        # new s is considered useless if isOver==True
        while True:
            batch_exp = [self.sample_one() for _ in range(self.batch_size)]

            #def view_state(state, next_state):
                #""" for debugging state representation"""
                #r = np.concatenate([state[:,:,k] for k in range(self.history_len)], axis=1)
                #r2 = np.concatenate([next_state[:,:,k] for k in range(self.history_len)], axis=1)
                #r = np.concatenate([r, r2], axis=0)
                #print r.shape
                #cv2.imshow("state", r)
                #cv2.waitKey()
            #exp = batch_exp[0]
            #print("Act: ", exp[3], " reward:", exp[2], " isOver: ", exp[4])
            #if exp[2] or exp[4]:
                #view_state(exp[0], exp[1])

            yield self._process_batch(batch_exp)
            for _ in range(self.new_experience_per_step):
                self._populate_exp()

    def sample_one(self):
        """ return the transition tuple for
            [idx, idx+history_len] -> [idx+1, idx+1+history_len]
            it's the transition from state idx+history_len-1 to state idx+history_len
        """
        # look for a state to start with
        # when x.isOver==True, (x+1).state is of a different episode
        idx = self.rng.randint(len(self.mem) - self.history_len - 1)
        start_idx = idx + self.history_len - 1

        def concat(idx):
            v = [self.mem[x].state for x in range(idx, idx+self.history_len)]
            return np.concatenate(v, axis=2)
        state = concat(idx)
        next_state = concat(idx + 1)
        reward = self.mem[start_idx].reward
        action = self.mem[start_idx].action
        isOver = self.mem[start_idx].isOver

        # zero-fill state before starting
        zero_fill = False
        for k in range(1, self.history_len):
            if self.mem[start_idx-k].isOver:
                zero_fill = True
            if zero_fill:
                state[:,:,-k-1] = 0
                if k + 2 <= self.history_len:
                    next_state[:,:,-k-2] = 0
        return (state, next_state, reward, action, isOver)

    def _process_batch(self, batch_exp):
        state = np.array([e[0] for e in batch_exp])
        next_state = np.array([e[1] for e in batch_exp])
        reward = np.array([e[2] for e in batch_exp])
        action = np.array([e[3] for e in batch_exp])
        isOver = np.array([e[4] for e in batch_exp])
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
    from tensorpack.dataflow.dataset import AtariPlayer
    import sys
    predictor = lambda x: np.array([1,1,1,1])
    predictor.initialized = False
    player = AtariPlayer(sys.argv[1], viz=0, frame_skip=10, height_range=(36, 204))
    E = ExpReplay(predictor,
            player=player,
            num_actions=player.get_num_actions(),
            populate_size=1001,
            history_len=4)
    E.init_memory()

    for k in E.get_data():
        import IPython as IP;
        IP.embed(config=IP.terminal.ipapp.load_default_config())
        pass
        #import IPython;
        #IPython.embed(config=IPython.terminal.ipapp.load_default_config())
        #break
