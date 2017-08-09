#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: envbase.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>


from abc import abstractmethod, ABCMeta
from collections import defaultdict
import six
from ..utils.utils import get_rng

__all__ = ['RLEnvironment', 'ProxyPlayer',
           'DiscreteActionSpace']


@six.add_metaclass(ABCMeta)
class RLEnvironment(object):
    """ Base class of RL environment. """

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
        Perform an action. Will automatically start a new episode if isOver==True

        Args:
            act: the action
        Returns:
            tuple: (reward, isOver)
        """

    def restart_episode(self):
        """ Start a new episode, even if the current hasn't ended """
        raise NotImplementedError()

    def finish_episode(self):
        """ Get called when an episode finished"""
        pass

    def get_action_space(self):
        """ Returns:
            :class:`ActionSpace` """
        raise NotImplementedError()

    def reset_stat(self):
        """ Reset all statistics counter"""
        self.stats = defaultdict(list)

    def play_one_episode(self, func, stat='score'):
        """ Play one episode for eval.

        Args:
            func: the policy function. Takes a state and returns an action.
            stat: a key or list of keys in stats to return.
        Returns:
            the stat(s) after running this episode
        """
        if not isinstance(stat, list):
            stat = [stat]
        while True:
            s = self.current_state()
            act = func(s)
            r, isOver = self.action(act)
            # print r
            if isOver:
                s = [self.stats[k] for k in stat]
                self.reset_stat()
                return s if len(s) > 1 else s[0]


class ActionSpace(object):

    def __init__(self):
        self.rng = get_rng(self)

    @abstractmethod
    def sample(self):
        pass

    def num_actions(self):
        raise NotImplementedError()


class DiscreteActionSpace(ActionSpace):

    def __init__(self, num):
        super(DiscreteActionSpace, self).__init__()
        self.num = num

    def sample(self):
        return self.rng.randint(self.num)

    def num_actions(self):
        return self.num

    def __repr__(self):
        return "DiscreteActionSpace({})".format(self.num)

    def __str__(self):
        return "DiscreteActionSpace({})".format(self.num)


class NaiveRLEnvironment(RLEnvironment):
    """ For testing only"""

    def __init__(self):
        self.k = 0

    def current_state(self):
        self.k += 1
        return self.k

    def action(self, act):
        self.k = act
        return (self.k, self.k > 10)


class ProxyPlayer(RLEnvironment):
    """ Serve as a proxy to another player """

    def __init__(self, player):
        self.player = player

    def reset_stat(self):
        self.player.reset_stat()

    def current_state(self):
        return self.player.current_state()

    def action(self, act):
        return self.player.action(act)

    @property
    def stats(self):
        return self.player.stats

    def restart_episode(self):
        self.player.restart_episode()

    def finish_episode(self):
        self.player.finish_episode()

    def get_action_space(self):
        return self.player.get_action_space()
