#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: common.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>


from collections import deque
from .envbase import ProxyPlayer

__all__ = ['PreventStuckPlayer', 'LimitLengthPlayer', 'AutoRestartPlayer',
           'MapPlayerState']


class PreventStuckPlayer(ProxyPlayer):
    """ Prevent the player from getting stuck (repeating a no-op)
    by inserting a different action. Useful in games such as Atari Breakout
    where the agent needs to press the 'start' button to start playing.

    It does auto-reset, but doesn't auto-restart the underlying player.
    """
    # TODO hash the state as well?

    def __init__(self, player, nr_repeat, action):
        """
        Args:
            nr_repeat (int): trigger the 'action' after this many of repeated action.
            action: the action to be triggered to get out of stuck.
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
    """ Limit the total number of actions in an episode.
        Will restart the underlying player on timeout.
    """

    def __init__(self, player, limit):
        """
        Args:
            limit(int): the time limit
        """
        super(LimitLengthPlayer, self).__init__(player)
        self.limit = limit
        self.cnt = 0

    def action(self, act):
        r, isOver = self.player.action(act)
        self.cnt += 1
        if self.cnt >= self.limit:
            isOver = True
            self.finish_episode()
            self.restart_episode()
        if isOver:
            self.cnt = 0
        return (r, isOver)

    def restart_episode(self):
        self.player.restart_episode()
        self.cnt = 0


class AutoRestartPlayer(ProxyPlayer):
    """ Auto-restart the player on episode ends,
        in case some player wasn't designed to do so.
    """

    def action(self, act):
        r, isOver = self.player.action(act)
        if isOver:
            self.player.finish_episode()
            self.player.restart_episode()
        return r, isOver


class MapPlayerState(ProxyPlayer):
    """ Map the state of the underlying player by a function. """

    def __init__(self, player, func):
        """
        Args:
            func: takes the old state and return a new state.
        """
        super(MapPlayerState, self).__init__(player)
        self.func = func

    def current_state(self):
        return self.func(self.player.current_state())
