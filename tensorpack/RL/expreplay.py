#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: expreplay.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import numpy as np
from collections import deque, namedtuple
import threading
import six
from six.moves import queue

from ..dataflow import DataFlow
from ..utils import logger, get_tqdm, get_rng
from ..utils.concurrency import LoopThread
from ..callbacks.base import Callback

__all__ = ['ExpReplay']

Experience = namedtuple('Experience',
                        ['state', 'action', 'reward', 'isOver'])


class ExpReplay(DataFlow, Callback):
    """
    Implement experience replay in the paper
    `Human-level control through deep reinforcement learning
    <http://www.nature.com/nature/journal/v518/n7540/full/nature14236.html>`_.

    This implementation provides the interface as a :class:`DataFlow`.
    This DataFlow is __not__ fork-safe (thus doesn't support multiprocess prefetching).

    This implementation only works with Q-learning. It assumes that state is
    batch-able, and the network takes batched inputs.
    """

    def __init__(self,
                 predictor_io_names,
                 player,
                 batch_size=32,
                 memory_size=1e6,
                 init_memory_size=50000,
                 exploration=1,
                 end_exploration=0.1,
                 exploration_epoch_anneal=0.002,
                 reward_clip=None,
                 update_frequency=1,
                 history_len=1
                 ):
        """
        Args:
            predictor_io_names (tuple of list of str): input/output names to
                predict Q value from state.
            player (RLEnvironment): the player.
            history_len (int): length of history frames to concat. Zero-filled
                initial frames.
            update_frequency (int): number of new transitions to add to memory
                after sampling a batch of transitions for training.
        """
        init_memory_size = int(init_memory_size)

        for k, v in locals().items():
            if k != 'self':
                setattr(self, k, v)
        self.num_actions = player.get_action_space().num_actions()
        logger.info("Number of Legal actions: {}".format(self.num_actions))
        self.mem = deque(maxlen=int(memory_size))
        self.rng = get_rng(self)
        self._init_memory_flag = threading.Event()  # tell if memory has been initialized
        self._predictor_io_names = predictor_io_names

    def _init_memory(self):
        logger.info("Populating replay memory...")

        # fill some for the history
        old_exploration = self.exploration
        self.exploration = 1
        for k in range(self.history_len):
            self._populate_exp()
        self.exploration = old_exploration

        with get_tqdm(total=self.init_memory_size) as pbar:
            while len(self.mem) < self.init_memory_size:
                self._populate_exp()
                pbar.update()
        self._init_memory_flag.set()

    def _populate_exp(self):
        """ populate a transition by epsilon-greedy"""
        # if len(self.mem):
        # from copy import deepcopy  # quickly fill the memory for debug
        # self.mem.append(deepcopy(self.mem[0]))
        # return
        old_s = self.player.current_state()
        if self.rng.rand() <= self.exploration:
            act = self.rng.choice(range(self.num_actions))
        else:
            # build a history state
            # XXX assume a state can be representated by one tensor
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
            # XXX assume batched network
            q_values = self.predictor([[ss]])[0][0]
            act = np.argmax(q_values)
        reward, isOver = self.player.action(act)
        if self.reward_clip:
            reward = np.clip(reward, self.reward_clip[0], self.reward_clip[1])
        self.mem.append(Experience(old_s, act, reward, isOver))

    def get_data(self):
        self._init_memory_flag.wait()
        # new s is considered useless if isOver==True
        while True:
            batch_exp = [self._sample_one() for _ in range(self.batch_size)]

            # import cv2  # for debug
            # def view_state(state, next_state):
            # """ for debugging state representation"""
            #     r = np.concatenate([state[:,:,k] for k in range(self.history_len)], axis=1)
            #     r2 = np.concatenate([next_state[:,:,k] for k in range(self.history_len)], axis=1)
            #     r = np.concatenate([r, r2], axis=0)
            #     print r.shape
            #     cv2.imshow("state", r)
            #     cv2.waitKey()
            # exp = batch_exp[0]
            # print("Act: ", exp[3], " reward:", exp[2], " isOver: ", exp[4])
            # if exp[2] or exp[4]:
            #     view_state(exp[0], exp[1])

            yield self._process_batch(batch_exp)
            self._populate_job_queue.put(1)

    def _sample_one(self):
        """ return the transition tuple for
            [idx, idx+history_len) -> [idx+1, idx+1+history_len)
            it's the transition from state idx+history_len-1 to state idx+history_len
        """
        # look for a state to start with
        # when x.isOver==True, (x+1).state is of a different episode
        idx = self.rng.randint(len(self.mem) - self.history_len - 1)

        samples = [self.mem[k] for k in range(idx, idx + self.history_len + 1)]

        def concat(idx):
            v = [x.state for x in samples[idx:idx + self.history_len]]
            return np.concatenate(v, axis=2)
        state = concat(0)
        next_state = concat(1)
        start_mem = samples[-2]
        reward, action, isOver = start_mem.reward, start_mem.action, start_mem.isOver

        start_idx = self.history_len - 1

        # zero-fill state before starting
        zero_fill = False
        for k in range(1, self.history_len):
            if samples[start_idx - k].isOver:
                zero_fill = True
            if zero_fill:
                state[:, :, -k - 1] = 0
                if k + 2 <= self.history_len:
                    next_state[:, :, -k - 2] = 0
        return (state, next_state, reward, action, isOver)

    def _process_batch(self, batch_exp):
        state = np.array([e[0] for e in batch_exp])
        next_state = np.array([e[1] for e in batch_exp])
        reward = np.array([e[2] for e in batch_exp])
        action = np.array([e[3] for e in batch_exp], dtype='int8')
        isOver = np.array([e[4] for e in batch_exp], dtype='bool')
        return [state, action, reward, next_state, isOver]

    def _setup_graph(self):
        self.predictor = self.trainer.get_predict_func(*self._predictor_io_names)

    # Callback-related:
    def _before_train(self):
        # spawn a separate thread to run policy, can speed up 1.3x
        self._populate_job_queue = queue.Queue(maxsize=1)

        def populate_job_func():
            self._populate_job_queue.get()
            with self.trainer.sess.as_default():
                for _ in range(self.update_frequency):
                    self._populate_exp()
        self._populate_job_th = LoopThread(populate_job_func, False)
        self._populate_job_th.start()

        self._init_memory()

    def _trigger_epoch(self):
        if self.exploration > self.end_exploration:
            self.exploration -= self.exploration_epoch_anneal
            logger.info("Exploration changed to {}".format(self.exploration))
        # log player statistics
        stats = self.player.stats
        for k, v in six.iteritems(stats):
            try:
                mean, max = np.mean(v), np.max(v)
                self.trainer.add_scalar_summary('expreplay/mean_' + k, mean)
                self.trainer.add_scalar_summary('expreplay/max_' + k, max)
            except:
                pass
        self.player.reset_stat()


if __name__ == '__main__':
    from .atari import AtariPlayer
    import sys

    def predictor(x):
        np.array([1, 1, 1, 1])
    player = AtariPlayer(sys.argv[1], viz=0, frame_skip=10, height_range=(36, 204))
    E = ExpReplay(predictor,
                  player=player,
                  num_actions=player.get_action_space().num_actions(),
                  populate_size=1001,
                  history_len=4)
    E._init_memory()

    for k in E.get_data():
        import IPython as IP
        IP.embed(config=IP.terminal.ipapp.load_default_config())
        pass
        # import IPython;
        # IPython.embed(config=IPython.terminal.ipapp.load_default_config())
        # break
