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

