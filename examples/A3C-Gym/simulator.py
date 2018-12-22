#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: simulator.py
# Author: Yuxin Wu

import multiprocessing as mp
import os
import threading
import time
from abc import ABCMeta, abstractmethod
from collections import defaultdict
import six
import zmq
from six.moves import queue

from tensorpack.utils import logger
from tensorpack.utils.concurrency import LoopThread, enable_death_signal, ensure_proc_terminate
from tensorpack.utils.serialize import dumps, loads

__all__ = ['SimulatorProcess', 'SimulatorMaster',
           'SimulatorProcessStateExchange',
           'TransitionExperience']


class TransitionExperience(object):
    """ A transition of state, or experience"""

    def __init__(self, state, action, reward, **kwargs):
        """ kwargs: whatever other attribute you want to save"""
        self.state = state
        self.action = action
        self.reward = reward
        for k, v in six.iteritems(kwargs):
            setattr(self, k, v)


@six.add_metaclass(ABCMeta)
class SimulatorProcessBase(mp.Process):
    def __init__(self, idx):
        super(SimulatorProcessBase, self).__init__()
        self.idx = int(idx)
        self.name = u'simulator-{}'.format(self.idx)
        self.identity = self.name.encode('utf-8')

    @abstractmethod
    def _build_player(self):
        pass


class SimulatorProcessStateExchange(SimulatorProcessBase):
    """
    A process that simulates a player and communicates to master to
    send states and receive the next action
    """

    def __init__(self, idx, pipe_c2s, pipe_s2c):
        """
        Args:
            idx: idx of this process
            pipe_c2s, pipe_s2c (str): name of the pipe
        """
        super(SimulatorProcessStateExchange, self).__init__(idx)
        self.c2s = pipe_c2s
        self.s2c = pipe_s2c

    def run(self):
        enable_death_signal()
        player = self._build_player()
        context = zmq.Context()
        c2s_socket = context.socket(zmq.PUSH)
        c2s_socket.setsockopt(zmq.IDENTITY, self.identity)
        c2s_socket.set_hwm(2)
        c2s_socket.connect(self.c2s)

        s2c_socket = context.socket(zmq.DEALER)
        s2c_socket.setsockopt(zmq.IDENTITY, self.identity)
        s2c_socket.connect(self.s2c)

        state = player.reset()
        reward, isOver = 0, False
        while True:
            # after taking the last action, get to this state and get this reward/isOver.
            # If isOver, get to the next-episode state immediately.
            # This tuple is not the same as the one put into the memory buffer
            c2s_socket.send(dumps(
                (self.identity, state, reward, isOver)),
                copy=False)
            action = loads(s2c_socket.recv(copy=False))
            state, reward, isOver, _ = player.step(action)
            if isOver:
                state = player.reset()


# compatibility
SimulatorProcess = SimulatorProcessStateExchange


class SimulatorMaster(threading.Thread):
    """ A base thread to communicate with all StateExchangeSimulatorProcess.
        It should produce action for each simulator, as well as
        defining callbacks when a transition or an episode is finished.
    """
    class ClientState(object):
        def __init__(self):
            self.memory = []    # list of Experience
            self.ident = None

    def __init__(self, pipe_c2s, pipe_s2c):
        super(SimulatorMaster, self).__init__()
        assert os.name != 'nt', "Doesn't support windows!"
        self.daemon = True
        self.name = 'SimulatorMaster'

        self.context = zmq.Context()

        self.c2s_socket = self.context.socket(zmq.PULL)
        self.c2s_socket.bind(pipe_c2s)
        self.c2s_socket.set_hwm(10)
        self.s2c_socket = self.context.socket(zmq.ROUTER)
        self.s2c_socket.bind(pipe_s2c)
        self.s2c_socket.set_hwm(10)

        # queueing messages to client
        self.send_queue = queue.Queue(maxsize=100)

        def f():
            msg = self.send_queue.get()
            self.s2c_socket.send_multipart(msg, copy=False)
        self.send_thread = LoopThread(f)
        self.send_thread.daemon = True
        self.send_thread.start()

        # make sure socket get closed at the end
        def clean_context(soks, context):
            for s in soks:
                s.close()
            context.term()
        import atexit
        atexit.register(clean_context, [self.c2s_socket, self.s2c_socket], self.context)

    def run(self):
        self.clients = defaultdict(self.ClientState)
        try:
            while True:
                msg = loads(self.c2s_socket.recv(copy=False))
                ident, state, reward, isOver = msg
                client = self.clients[ident]
                if client.ident is None:
                    client.ident = ident
                # maybe check history and warn about dead client?
                self._process_msg(client, state, reward, isOver)
        except zmq.ContextTerminated:
            logger.info("[Simulator] Context was terminated.")

    def __del__(self):
        self.context.destroy(linger=0)


if __name__ == '__main__':
    import random
    import gym

    class NaiveSimulator(SimulatorProcess):
        def _build_player(self):
            return gym.make('Breakout-v0')

    class NaiveActioner(SimulatorMaster):
        def _get_action(self, state):
            time.sleep(1)
            return random.randint(1, 3)

        def _on_episode_over(self, client):
            # print("Over: ", client.memory)
            client.memory = []
            client.state = 0

    name = 'ipc://@whatever'
    procs = [NaiveSimulator(k, name) for k in range(10)]
    [k.start() for k in procs]

    th = NaiveActioner(name)
    ensure_proc_terminate(procs)
    th.start()

    time.sleep(100)
