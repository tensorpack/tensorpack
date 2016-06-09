#!/usr/bin/env python2
# -*- coding: utf-8 -*-
# File: simulator.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import multiprocessing
import time
import threading
import weakref
from abc import abstractmethod, ABCMeta
from collections import defaultdict, namedtuple
import numpy as np
import six
from six.moves import queue

from ..utils.timer import *
from ..utils.serialize import *
from ..utils.concurrency import *

__all__ = ['SimulatorProcess', 'SimulatorMaster']

try:
    import zmq
except ImportError:
    logger.warn("Error in 'import zmq'. RL simulator won't be available.")
    __all__ = []


class SimulatorProcess(multiprocessing.Process):
    """ A process that simulates a player """
    __metaclass__ = ABCMeta

    def __init__(self, idx, pipe_c2s, pipe_s2c):
        """
        :param idx: idx of this process
        """
        super(SimulatorProcess, self).__init__()
        self.idx = int(idx)
        self.c2s = pipe_c2s
        self.s2c = pipe_s2c

    def run(self):
        player = self._build_player()
        context = zmq.Context()
        c2s_socket = context.socket(zmq.DEALER)
        c2s_socket.identity = 'simulator-{}'.format(self.idx)
        c2s_socket.set_hwm(2)
        c2s_socket.connect(self.c2s)

        s2c_socket = context.socket(zmq.DEALER)
        s2c_socket.identity = 'simulator-{}'.format(self.idx)
        #s2c_socket.set_hwm(5)
        s2c_socket.connect(self.s2c)

        #cnt = 0
        while True:
            state = player.current_state()
            c2s_socket.send(dumps(state), copy=False)
            #with total_timer('client recv_action'):
            data = s2c_socket.recv(copy=False)
            action = loads(data)
            reward, isOver = player.action(action)
            c2s_socket.send(dumps((reward, isOver)), copy=False)
            #with total_timer('client recv_ack'):
            ACK = s2c_socket.recv(copy=False)
            #cnt += 1
            #if cnt % 100 == 0:
                #print_total_timer()

    @abstractmethod
    def _build_player(self):
        pass

class SimulatorMaster(threading.Thread):
    """ A base thread to communicate with all simulator processes.
        It should produce action for each simulator, as well as
        defining callbacks when a transition or an episode is finished.
    """
    __metaclass__ = ABCMeta

    class ClientState(object):
        def __init__(self):
            self.protocol_state = 0 # state in communication
            self.memory = []    # list of Experience

    class Experience(object):
        """ A transition of state, or experience"""
        def __init__(self, state, action, reward, **kwargs):
            """ kwargs: whatever other attribute you want to save"""
            self.state = state
            self.action = action
            self.reward = reward
            for k, v in six.iteritems(kwargs):
                setattr(self, k, v)

    def __init__(self, pipe_c2s, pipe_s2c):
        super(SimulatorMaster, self).__init__()
        self.context = zmq.Context()

        self.c2s_socket = self.context.socket(zmq.ROUTER)
        self.c2s_socket.bind(pipe_c2s)

        self.s2c_socket = self.context.socket(zmq.ROUTER)
        self.s2c_socket.bind(pipe_s2c)

        self.socket_lock = threading.Lock()
        self.daemon = True

        # queueing messages to client
        self.send_queue = queue.Queue(maxsize=100)
        self.send_thread = LoopThread(lambda:
                self.s2c_socket.send_multipart(self.send_queue.get()))
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
        while True:
            ident, msg = self.c2s_socket.recv_multipart()
            client = self.clients[ident]
            client.protocol_state = 1 - client.protocol_state   # first flip the state
            if not client.protocol_state == 0:   # state-action
                state = loads(msg)
                self._on_state(state, ident)
            else:       # reward-response
                reward, isOver = loads(msg)
                client.memory[-1].reward = reward
                if isOver:
                    self._on_episode_over(ident)
                else:
                    self._on_datapoint(ident)
                self.send_queue.put([ident, 'Thanks'])  # just an ACK

    @abstractmethod
    def _on_state(self, state, ident):
        """response to state sent by ident. Preferrably an async call"""

    @abstractmethod
    def _on_episode_over(self, client):
        """ callback when the client just finished an episode.
            You may want to clear the client's memory in this callback.
        """

    def _on_datapoint(self, client):
        """ callback when the client just finished a transition
        """

    def __del__(self):
        self.socket.close()
        self.context.term()

if __name__ == '__main__':
    import random
    from tensorpack.RL import NaiveRLEnvironment
    class NaiveSimulator(SimulatorProcess):
        def _build_player(self):
            return NaiveRLEnvironment()
    class NaiveActioner(SimulatorActioner):
        def _get_action(self, state):
            time.sleep(1)
            return random.randint(1, 12)
        def _on_episode_over(self, client):
            #print("Over: ", client.memory)
            client.memory = []
            client.state = 0

    name = 'ipc://whatever'
    procs = [NaiveSimulator(k, name) for k in range(10)]
    [k.start() for k in procs]

    th = NaiveActioner(name)
    ensure_proc_terminate(procs)
    th.start()

    import time
    time.sleep(100)

