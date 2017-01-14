#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: history.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import numpy as np
from collections import deque
from .envbase import ProxyPlayer

__all__ = ['HistoryFramePlayer']


class HistoryFramePlayer(ProxyPlayer):
    """ Include history frames in state, or use black images.
        It assumes the underlying player will do auto-restart.
    """

    def __init__(self, player, hist_len):
        """
        Args:
            hist_len (int): total length of the state, including the current
                and `hist_len-1` history.
        """
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
        assert len(zeros) == self.history.maxlen
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
