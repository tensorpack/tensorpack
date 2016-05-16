#!/usr/bin/env python2
# -*- coding: utf-8 -*-
# File: exp_replay.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

from tensorpack.dataflow import *
from tensorpack.dataflow.dataset import AtariDriver, AtariPlayer
from tensorpack.utils import *
from tqdm import tqdm
import random
import numpy as np
import cv2

from collections import deque, namedtuple

Experience = namedtuple('Experience',
        ['state', 'action', 'reward', 'next', 'isOver'])

def view_state(state):
    r = np.concatenate([state[:,:,k] for k in range(state.shape[2])], axis=1)
    print r.shape
    cv2.imshow("state", r)
    cv2.waitKey()

class AtariExpReplay(DataFlow):
    """
    Implement experience replay
    """
    def __init__(self,
            predictor,
            player,
            memory_size=1e6,
            batch_size=32,
            populate_size=50000,
            exploration=1):
        """
        :param predictor: callabale. called with a state, return a distribution
        """
        for k, v in locals().items():
            if k != 'self':
                setattr(self, k, v)
        self.num_actions = self.player.driver.get_num_actions()
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
        _, reward, isOver = self.player.action(act)
        reward = np.clip(reward, -1, 2)
        s = self.player.current_state()

        #print act, reward
        #view_state(s)

        self.mem.append(Experience(old_s, act, reward, s, isOver))

    def get_data(self):
        while True:
            idxs = self.rng.randint(len(self.mem), size=self.batch_size)
            batch_exp = [self.mem[k] for k in idxs]
            yield self._process_batch(batch_exp)
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

if __name__ == '__main__':
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
