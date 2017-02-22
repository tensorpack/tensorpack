#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: gymenv.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>


import time
import threading

from ..utils.fs import mkdir_p
from ..utils.stats import StatCounter
from .envbase import RLEnvironment, DiscreteActionSpace


__all__ = ['GymEnv']
_ENV_LOCK = threading.Lock()


class GymEnv(RLEnvironment):
    """
    An OpenAI/gym wrapper. Can optionally auto restart.
    Only support discrete action space for now.
    """

    def __init__(self, name, dumpdir=None, viz=False, auto_restart=True):
        """
        Args:
            name (str): the gym environment name.
            dumpdir (str): the directory to dump recordings to.
            viz (bool): whether to start visualization.
            auto_restart (bool): whether to restart after episode ends.
        """
        with _ENV_LOCK:
            self.gymenv = gym.make(name)
        if dumpdir:
            mkdir_p(dumpdir)
            self.gymenv = gym.wrappers.Monitor(self.gymenv, dumpdir)
        self.use_dir = dumpdir

        self.reset_stat()
        self.rwd_counter = StatCounter()
        self.restart_episode()
        self.auto_restart = auto_restart
        self.viz = viz

    def restart_episode(self):
        self.rwd_counter.reset()
        self._ob = self.gymenv.reset()

    def finish_episode(self):
        self.stats['score'].append(self.rwd_counter.sum)

    def current_state(self):
        if self.viz:
            self.gymenv.render()
            time.sleep(self.viz)
        return self._ob

    def action(self, act):
        self._ob, r, isOver, info = self.gymenv.step(act)
        self.rwd_counter.feed(r)
        if isOver:
            self.finish_episode()
            if self.auto_restart:
                self.restart_episode()
        return r, isOver

    def get_action_space(self):
        spc = self.gymenv.action_space
        assert isinstance(spc, gym.spaces.discrete.Discrete)
        return DiscreteActionSpace(spc.n)


try:
    import gym
    import gym.wrappers
    # TODO
    # gym.undo_logger_setup()
    # https://github.com/openai/gym/pull/199
    # not sure does it cause other problems
except ImportError:
    from ..utils.develop import create_dummy_class
    GymEnv = create_dummy_class('GymEnv', 'gym')    # noqa


if __name__ == '__main__':
    env = GymEnv('Breakout-v0', viz=0.1)
    num = env.get_action_space().num_actions()

    from ..utils import get_rng
    rng = get_rng(num)
    while True:
        act = rng.choice(range(num))
        # print act
        r, o = env.action(act)
        env.current_state()
        if r != 0 or o:
            print(r, o)
