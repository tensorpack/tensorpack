#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: common.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>


import numpy as np
from collections import deque
from .envbase import ProxyPlayer

__all__ = ['HistoryFramePlayer', 'PreventStuckPlayer', 'LimitLengthPlayer']

class HistoryFramePlayer(ProxyPlayer):
    """ Include history frames in state, or use black images"""
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

    def restart_episode(self):
        super(HistoryFramePlayer, self).restart_episode()
        self.history.clear()
        self.history.append(self.player.current_state())

class PreventStuckPlayer(ProxyPlayer):
    """ Prevent the player from getting stuck (repeating a no-op)
    by inserting a different action. Useful in games such as Atari Breakout
    where the agent needs to press the 'start' button to start playing.
    """
    # TODO hash the state as well?
    def __init__(self, player, nr_repeat, action):
        """
        :param nr_repeat: trigger the 'action' after this many of repeated action
        :param action: the action to be triggered to get out of stuck
        """
        super(PreventStuckPlayer, self).__init__(player)
        self.act_que = deque(maxlen=nr_repeat)
        self.trigger_action = action

    def action(self, act):
        self.act_que.append(act)
        if self.act_que.count(self.act_que[0]) == self.act_que.maxlen:
            act = self.trigger_action
        r, isOver = self.player.action(act)
        if isOver:
            self.act_que.clear()
        return (r, isOver)

    def restart_episode(self):
        super(PreventStuckPlayer, self).restart_episode()
        self.act_que.clear()

class LimitLengthPlayer(ProxyPlayer):
    """ Limit the total number of actions in an episode"""
    def __init__(self, player, limit):
        super(LimitLengthPlayer, self).__init__(player)
        self.limit = limit
        self.cnt = 0

    def action(self, act):
        r, isOver = self.player.action(act)
        self.cnt += 1
        if self.cnt >= self.limit:
            isOver = True
            self.player.restart_episode()
        if isOver:
            #print self.cnt, self.player.stats  # to see what limit is appropriate
            self.cnt = 0
        return (r, isOver)

    def restart_episode(self):
        super(LimitLengthPlayer, self).restart_episode()
        self.cnt = 0
