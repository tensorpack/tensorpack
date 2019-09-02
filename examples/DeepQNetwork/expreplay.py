# -*- coding: utf-8 -*-
# File: expreplay.py
# Author: Yuxin Wu

import copy
import itertools
import numpy as np
import threading
from collections import namedtuple
from six.moves import queue, range

from tensorpack.utils.concurrency import LoopThread, ShareSessionThread
from tensorpack.callbacks.base import Callback
from tensorpack.dataflow import DataFlow
from tensorpack.utils import logger, get_rng, get_tqdm
from tensorpack.utils.stats import StatCounter

__all__ = ['ExpReplay']

Experience = namedtuple('Experience',
                        ['state', 'action', 'reward', 'isOver'])


class ReplayMemory(object):
    def __init__(self, max_size, state_shape, history_len, dtype='uint8'):
        """
        Args:
            state_shape (tuple[int]): shape (without history) of state
            dtype: numpy dtype for the state
        """
        self.max_size = int(max_size)
        self.state_shape = state_shape
        assert len(state_shape) in [1, 2, 3], state_shape
        # self._output_shape = self.state_shape + (history_len + 1, )
        self.history_len = int(history_len)
        self.dtype = dtype

        all_state_shape = (self.max_size,) + state_shape
        logger.info("Creating experience replay buffer of {:.1f} GB ... "
                    "use a smaller buffer if you don't have enough CPU memory.".format(
                        np.prod(all_state_shape) / 1024.0**3))
        self.state = np.zeros(all_state_shape, dtype=self.dtype)
        self.action = np.zeros((self.max_size,), dtype='int32')
        self.reward = np.zeros((self.max_size,), dtype='float32')
        self.isOver = np.zeros((self.max_size,), dtype='bool')

        self._curr_size = 0
        self._curr_pos = 0

        self.writer_lock = threading.Lock()  # a lock to guard writing to the memory

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

    def sample(self, idx):
        """ return a tuple of (s,r,a,o),
            where s is of shape self._output_shape, which is
            [H, W, (hist_len+1) * channel] if input is (H, W, channel)"""
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
        # state: Hist+1,H,W,C
        for k in range(self.history_len - 2, -1, -1):
            if isOver[k]:
                state = copy.deepcopy(state)
                state[:k + 1].fill(0)
                break
        # move the first dim (history) to the last
        state = np.moveaxis(state, 0, -1)
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


class EnvRunner(object):
    """
    A class which is responsible for
    stepping the environment with epsilon-greedy,
    and fill the results to experience replay buffer.
    """
    def __init__(self, player, predictor, memory, history_len):
        """
        Args:
            player (gym.Env)
            predictor (callable): the model forward function which takes a
                state and returns the prediction.
            memory (ReplayMemory): the replay memory to store experience to.
            history_len (int):
        """
        self.player = player
        self.num_actions = player.action_space.n
        self.predictor = predictor
        self.memory = memory
        self.state_shape = memory.state_shape
        self.dtype = memory.dtype
        self.history_len = history_len

        self._current_episode = []
        self._current_ob = player.reset()
        self._current_game_score = StatCounter()  # store per-step reward
        self.total_scores = []  # store per-game total score

        self.rng = get_rng(self)

    def step(self, exploration):
        """
        Run the environment for one step.
        If the episode ends, store the entire episode to the replay memory.
        """
        old_s = self._current_ob
        if self.rng.rand() <= exploration:
            act = self.rng.choice(range(self.num_actions))
        else:
            history = self.recent_state()
            history.append(old_s)
            history = np.stack(history, axis=-1)  # state_shape + (Hist,)

            # assume batched network
            history = np.expand_dims(history, axis=0)
            q_values = self.predictor(history)[0][0]  # this is the bottleneck
            act = np.argmax(q_values)

        self._current_ob, reward, isOver, info = self.player.step(act)
        self._current_game_score.feed(reward)
        self._current_episode.append(Experience(old_s, act, reward, isOver))

        if isOver:
            flush_experience = True
            if 'ale.lives' in info:  # if running Atari, do something special
                if info['ale.lives'] != 0:
                    # only record score and flush experience
                    # when a whole game is over (not when an episode is over)
                    flush_experience = False
            self.player.reset()

            if flush_experience:
                self.total_scores.append(self._current_game_score.sum)
                self._current_game_score.reset()

                # Ensure that the whole episode of experience is continuous in the replay buffer
                with self.memory.writer_lock:
                    for exp in self._current_episode:
                        self.memory.append(exp)
                self._current_episode.clear()

    def recent_state(self):
        """
        Get the recent state (with stacked history) of the environment.

        Returns:
            a list of ``hist_len-1`` elements, each of shape ``self.state_shape``
        """
        expected_len = self.history_len - 1
        if len(self._current_episode) >= expected_len:
            return [k.state for k in self._current_episode[-expected_len:]]
        else:
            states = [np.zeros(self.state_shape, dtype=self.dtype)] * (expected_len - len(self._current_episode))
            states.extend([k.state for k in self._current_episode])
            return states


class EnvRunnerManager(object):
    """
    A class which manages a list of :class:`EnvRunner`.
    Its job is to execute them possibly in parallel and aggregate their results.
    """
    def __init__(self, env_runners, maximum_staleness):
        """
        Args:
            env_runners (list[EnvRunner]):
            maximum_staleness (int): when >1 environments run in parallel,
                the actual stepping of an environment may happen several steps
                after calls to `EnvRunnerManager.step()`, in order to achieve better throughput.
        """
        assert len(env_runners) > 0
        self._runners = env_runners

        if len(self._runners) > 1:
            # Only use threads when having >1 runners.
            self._populate_job_queue = queue.Queue(maxsize=maximum_staleness)
            self._threads = [self._create_simulator_thread(i) for i in range(len(self._runners))]
            for t in self._threads:
                t.start()

    def _create_simulator_thread(self, idx):
        # spawn a separate thread to run policy
        def populate_job_func():
            exp = self._populate_job_queue.get()
            self._runners[idx].step(exp)

        th = ShareSessionThread(LoopThread(populate_job_func, pausable=False))
        th.name = "SimulatorThread-{}".format(idx)
        return th

    def step(self, exploration):
        """
        Execute one step in any of the runners.
        """
        if len(self._runners) > 1:
            self._populate_job_queue.put(exploration)
        else:
            self._runners[0].step(exploration)

    def reset_stats(self):
        """
        Returns:
            mean, max: two stats of the runners, to be added to backend
        """
        scores = list(itertools.chain.from_iterable([v.total_scores for v in self._runners]))
        for v in self._runners:
            v.total_scores.clear()

        try:
            return np.mean(scores), np.max(scores)
        except Exception:
            logger.exception("Cannot compute total scores in EnvRunner.")
            return None, None


class ExpReplay(DataFlow, Callback):
    """
    Implement experience replay in the paper
    `Human-level control through deep reinforcement learning
    <http://www.nature.com/nature/journal/v518/n7540/full/nature14236.html>`_.

    This implementation provides the interface as a :class:`DataFlow`.
    This DataFlow is __not__ fork-safe (thus doesn't support multiprocess prefetching).

    It does the following:
    * Spawn `num_parallel_players` environment thread, each running an instance
      of the environment with epislon-greedy policy.
    * All environment instances writes their experiences to a shared replay
      memory buffer.
    * Produces batched samples by sampling the replay buffer. After producing
      each batch, it executes the environment instances by a total of
      `update_frequency` steps.

    This implementation assumes that state is batch-able, and the network takes batched inputs.
    """

    def __init__(self,
                 predictor_io_names,
                 get_player,
                 num_parallel_players,
                 state_shape,
                 batch_size,
                 memory_size, init_memory_size,
                 update_frequency, history_len,
                 state_dtype='uint8'):
        """
        Args:
            predictor_io_names (tuple of list of str): input/output names to
                predict Q value from state.
            get_player (-> gym.Env): a callable which returns a player.
            num_parallel_players (int): number of players to run in parallel.
                Standard DQN uses 1.
                Parallelism increases speed, but will affect the distribution of
                experiences in the replay buffer.
            state_shape (tuple):
            batch_size (int):
            memory_size (int):
            init_memory_size (int):
            update_frequency (int): number of new transitions to add to memory
                after sampling a batch of transitions for training.
            history_len (int): length of history frames to concat. Zero-filled
                initial frames.
            state_dtype (str):
        """
        assert len(state_shape) in [1, 2, 3], state_shape
        init_memory_size = int(init_memory_size)

        for k, v in locals().items():
            if k != 'self':
                setattr(self, k, v)
        self.exploration = 1.0  # default initial exploration

        self.rng = get_rng(self)
        self._init_memory_flag = threading.Event()  # tell if memory has been initialized

        self.mem = ReplayMemory(memory_size, state_shape, self.history_len, dtype=state_dtype)

    def _init_memory(self):
        logger.info("Populating replay memory with epsilon={} ...".format(self.exploration))

        with get_tqdm(total=self.init_memory_size) as pbar:
            while len(self.mem) < self.init_memory_size:
                self.runner.step(self.exploration)
                pbar.update()
        self._init_memory_flag.set()

    # quickly fill the memory for debug
    def _fake_init_memory(self):
        from copy import deepcopy
        with get_tqdm(total=self.init_memory_size) as pbar:
            while len(self.mem) < 5:
                self.runner.step(self.exploration)
                pbar.update()
            while len(self.mem) < self.init_memory_size:
                self.mem.append(deepcopy(self.mem._hist[0]))
                pbar.update()
        self._init_memory_flag.set()

    def _debug_sample(self, sample):
        import cv2

        def view_state(comb_state):
            # this function assumes comb_state is 3D
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

    def _process_batch(self, batch_exp):
        state = np.asarray([e[0] for e in batch_exp], dtype=self.state_dtype)
        reward = np.asarray([e[1] for e in batch_exp], dtype='float32')
        action = np.asarray([e[2] for e in batch_exp], dtype='int8')
        isOver = np.asarray([e[3] for e in batch_exp], dtype='bool')
        return [state, action, reward, isOver]

    # DataFlow method:
    def __iter__(self):
        # wait for memory to be initialized
        self._init_memory_flag.wait()

        while True:
            idx = self.rng.randint(
                0, len(self.mem) - self.history_len - 1,
                size=self.batch_size)
            batch_exp = [self.mem.sample(i) for i in idx]

            yield self._process_batch(batch_exp)

            # execute update_freq=4 new actions into memory, after each batch update
            for _ in range(self.update_frequency):
                self.runner.step(self.exploration)

    # Callback methods:
    def _setup_graph(self):
        self.predictor = self.trainer.get_predictor(*self.predictor_io_names)

    def _before_train(self):
        env_runners = [
            EnvRunner(self.get_player(), self.predictor, self.mem, self.history_len)
            for k in range(self.num_parallel_players)
        ]
        self.runner = EnvRunnerManager(env_runners, self.update_frequency * 2)
        self._init_memory()

    def _trigger(self):
        mean, max = self.runner.reset_stats()
        if mean is not None:
            self.trainer.monitors.put_scalar('expreplay/mean_score', mean)
            self.trainer.monitors.put_scalar('expreplay/max_score', max)
