#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: history.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import numpy as np
from collections import deque
from six.moves import range
from .envbase import ProxyPlayer

__all__ = ['HistoryFramePlayer']


class HistoryBuffer(object):
    def __init__(self, hist_len, concat_axis=2):
        self.buf = deque(maxlen=hist_len)
        self.concat_axis = concat_axis

    def push(self, s):
        self.buf.append(s)

    def clear(self):
        self.buf.clear()

    def get(self):
        difflen = self.buf.maxlen - len(self.buf)
        if difflen == 0:
            ret = self.buf
        else:
            zeros = [np.zeros_like(self.buf[0]) for k in range(difflen)]
            for k in self.buf:
                zeros.append(k)
            ret = zeros
        return np.concatenate(ret, axis=self.concat_axis)

    def __len__(self):
        return len(self.buf)

    @property
    def maxlen(self):
        return self.buf.maxlen


class HistoryFramePlayer(ProxyPlayer):
    """ Include history frames in state, or use black images.
        It assumes the underlying player will do auto-restart.

        Map the original frames into (H, W, HIST x channels).
        Oldest frames first.
    """

    def __init__(self, player, hist_len):
        """
        Args:
            hist_len (int): total length of the state, including the current
                and `hist_len-1` history.
        """
        super(HistoryFramePlayer, self).__init__(player)
        self.history = HistoryBuffer(hist_len, 2)

        s = self.player.current_state()
        self.history.push(s)

    def current_state(self):
        assert len(self.history) != 0
        return self.history.get()

    def action(self, act):
        r, isOver = self.player.action(act)
        s = self.player.current_state()
        self.history.push(s)

        if isOver:  # s would be a new episode
            self.history.clear()
            self.history.push(s)
        return (r, isOver)

    def restart_episode(self):
        super(HistoryFramePlayer, self).restart_episode()
        self.history.clear()
        self.history.push(self.player.current_state())
