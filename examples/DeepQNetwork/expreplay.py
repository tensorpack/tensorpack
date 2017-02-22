#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: expreplay.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import numpy as np
import copy
from collections import deque, namedtuple
import threading
import six
from six.moves import queue, range

from tensorpack.dataflow import DataFlow
from tensorpack.utils import logger, get_tqdm, get_rng
from tensorpack.utils.concurrency import LoopThread, ShareSessionThread
from tensorpack.callbacks.base import Callback

__all__ = ['ExpReplay']

Experience = namedtuple('Experience',
                        ['state', 'action', 'reward', 'isOver'])


class ReplayMemory(object):
    def __init__(self, max_size, state_shape, history_len):
        self.max_size = int(max_size)
        self.state_shape = state_shape
        self.history_len = int(history_len)

        self.state = np.zeros((self.max_size,) + state_shape, dtype='uint8')
        self.action = np.zeros((self.max_size,), dtype='int32')
        self.reward = np.zeros((self.max_size,), dtype='float32')
        self.isOver = np.zeros((self.max_size,), dtype='bool')

        self._curr_size = 0
        self._curr_pos = 0
        self._hist = deque(maxlen=history_len - 1)

    def append(self, exp):
        """
        Args:
            exp (Experience):
        """
        if self._curr_size < self.max_size:
            self._assign(self._curr_pos, exp)
            self._curr_pos = (self._curr_pos + 1) % self.max_size
            self._curr_size += 1
        else:
            self._assign(self._curr_pos, exp)
            self._curr_pos = (self._curr_pos + 1) % self.max_size
        if exp.isOver:
            self._hist.clear()
        else:
            self._hist.append(exp)

    def recent_state(self):
        """ return a list of (hist_len-1,) + STATE_SIZE """
        lst = list(self._hist)
        states = [np.zeros(self.state_shape, dtype='uint8')] * (self._hist.maxlen - len(lst))
        states.extend([k.state for k in lst])
        return states

    def sample(self, idx):
        """ return a tuple of (s,r,a,o),
            where s is of shape STATE_SIZE + (hist_len+1,)"""
        idx = (self._curr_pos + idx) % self._curr_size
        k = self.history_len + 1
        if idx + k <= self._curr_size:
            state = self.state[idx: idx + k]
            reward = self.reward[idx: idx + k]
            action = self.action[idx: idx + k]
            isOver = self.isOver[idx: idx + k]
        else:
            end = idx + k - self._curr_size
            state = self._slice(self.state, idx, end)
            reward = self._slice(self.reward, idx, end)
            action = self._slice(self.action, idx, end)
            isOver = self._slice(self.isOver, idx, end)
        ret = self._pad_sample(state, reward, action, isOver)
        return ret

    # the next_state is a different episode if current_state.isOver==True
    def _pad_sample(self, state, reward, action, isOver):
        for k in range(self.history_len - 2, -1, -1):
            if isOver[k]:
                state = copy.deepcopy(state)
                state[:k + 1].fill(0)
                break
        state = state.transpose(1, 2, 0)
        return (state, reward[-2], action[-2], isOver[-2])

    def _slice(self, arr, start, end):
        s1 = arr[start:]
        s2 = arr[:end]
        return np.concatenate((s1, s2), axis=0)

    def __len__(self):
        return self._curr_size

    def _assign(self, pos, exp):
        self.state[pos] = exp.state
        self.reward[pos] = exp.reward
        self.action[pos] = exp.action
        self.isOver[pos] = exp.isOver


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
                 state_shape,
                 batch_size,
                 memory_size, init_memory_size,
                 exploration, end_exploration, exploration_epoch_anneal,
                 update_frequency, history_len):
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

        self.rng = get_rng(self)
        self._init_memory_flag = threading.Event()  # tell if memory has been initialized

        # TODO just use a semaphore?
        # a queue to receive notifications to populate memory
        self._populate_job_queue = queue.Queue(maxsize=5)

        self.mem = ReplayMemory(memory_size, state_shape, history_len)

    def get_simulator_thread(self):
        # spawn a separate thread to run policy
        def populate_job_func():
            self._populate_job_queue.get()
            for _ in range(self.update_frequency):
                self._populate_exp()
        th = ShareSessionThread(LoopThread(populate_job_func, pausable=False))
        th.name = "SimulatorThread"
        return th

    def _init_memory(self):
        logger.info("Populating replay memory with epsilon={} ...".format(self.exploration))

        with get_tqdm(total=self.init_memory_size) as pbar:
            while len(self.mem) < self.init_memory_size:
                self._populate_exp()
                pbar.update()
        self._init_memory_flag.set()

    def _populate_exp(self):
        """ populate a transition by epsilon-greedy"""
        # if len(self.mem) > 4 and not self._init_memory_flag.is_set():
        #     from copy import deepcopy  # quickly fill the memory for debug
        #     self.mem.append(deepcopy(self.mem._hist[0]))
        #     return
        old_s = self.player.current_state()
        if self.rng.rand() <= self.exploration or len(self.mem) < 5:
            act = self.rng.choice(range(self.num_actions))
        else:
            # build a history state
            history = self.mem.recent_state()
            history.append(old_s)
            history = np.stack(history, axis=2)

            # assume batched network
            q_values = self.predictor([[history]])[0][0]  # this is the bottleneck
            act = np.argmax(q_values)
        reward, isOver = self.player.action(act)
        self.mem.append(Experience(old_s, act, reward, isOver))

    def debug_sample(self, sample):
        import cv2

        def view_state(comb_state):
            state = comb_state[:, :, :-1]
            next_state = comb_state[:, :, 1:]
            r = np.concatenate([state[:, :, k] for k in range(self.history_len)], axis=1)
            r2 = np.concatenate([next_state[:, :, k] for k in range(self.history_len)], axis=1)
            r = np.concatenate([r, r2], axis=0)
            cv2.imshow("state", r)
            cv2.waitKey()
        print("Act: ", sample[2], " reward:", sample[1], " isOver: ", sample[3])
        if sample[1] or sample[3]:
            view_state(sample[0])

    def get_data(self):
        # wait for memory to be initialized
        self._init_memory_flag.wait()

        while True:
            idx = self.rng.randint(
                self._populate_job_queue.maxsize * self.update_frequency,
                len(self.mem) - self.history_len - 1,
                size=self.batch_size)
            batch_exp = [self.mem.sample(i) for i in idx]

            yield self._process_batch(batch_exp)
            self._populate_job_queue.put(1)

    def _process_batch(self, batch_exp):
        state = np.asarray([e[0] for e in batch_exp], dtype='uint8')
        reward = np.asarray([e[1] for e in batch_exp], dtype='float32')
        action = np.asarray([e[2] for e in batch_exp], dtype='int8')
        isOver = np.asarray([e[3] for e in batch_exp], dtype='bool')
        return [state, action, reward, isOver]

    def _setup_graph(self):
        self.predictor = self.trainer.get_predictor(*self.predictor_io_names)

    def _before_train(self):
        self._init_memory()
        self._simulator_th = self.get_simulator_thread()
        self._simulator_th.start()

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
