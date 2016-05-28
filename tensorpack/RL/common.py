#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: common.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>


import numpy as np
from collections import deque
from .envbase import ProxyPlayer

__all__ = ['HistoryFramePlayer']

class HistoryFramePlayer(ProxyPlayer):
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

class AvoidNoOpPlayer(ProxyPlayer):
    pass # TODO
