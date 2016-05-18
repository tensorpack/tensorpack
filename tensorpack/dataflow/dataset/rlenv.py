#!/usr/bin/env python2
# -*- coding: utf-8 -*-
# File: rlenv.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

from abc import abstractmethod, ABCMeta

__all__ = ['RLEnvironment']

class RLEnvironment(object):
    __meta__ = ABCMeta

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
