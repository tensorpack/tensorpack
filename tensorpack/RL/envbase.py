#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: envbase.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>


from abc import abstractmethod, ABCMeta
from collections import defaultdict

__all__ = ['RLEnvironment', 'NaiveRLEnvironment', 'ProxyPlayer']

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
        Perform an action. Will automatically start a new episode if isOver==True
        :params act: the action
        :returns: (reward, isOver)
        """

    def restart_episode(self):
        """ Start a new episode, even if the current hasn't ended """
        raise NotImplementedError()

    def get_stat(self):
        """
        return a dict of statistics (e.g., score) for all the episodes since last call to reset_stat
        """
        return {}

    def reset_stat(self):
        """ reset the statistics counter"""
        self.stats = defaultdict(list)

    def play_one_episode(self, func, stat='score'):
        """ play one episode for eval.
            :params func: call with the state and return an action
            :returns: the score of this episode
        """
        while True:
            s = self.current_state()
            act = func(s)
            r, isOver = self.action(act)
            if isOver:
                s = self.stats[stat]
                self.reset_stat()
                return s

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
    def restart_episode(self):
        pass

class ProxyPlayer(RLEnvironment):
    """ Serve as a proxy another player """
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

    @property
    def stats(self):
        return self.player.stats

    def restart_episode(self):
        self.player.restart_episode()
