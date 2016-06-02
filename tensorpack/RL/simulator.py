#!/usr/bin/env python2
# -*- coding: utf-8 -*-
# File: simulator.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import multiprocessing
import threading
import weakref
from abc import abstractmethod, ABCMeta
from collections import defaultdict, namedtuple

from tensorpack.utils.serialize import *
from tensorpack.utils.concurrency import *

__all__ = ['SimulatorProcess', 'SimulatorMaster']

try:
    import zmq
except ImportError:
    logger.warn("Error in 'import zmq'. RL simulator won't be available.")
    __all__ = []


class SimulatorProcess(multiprocessing.Process):
    """ A process that simulates a player """
    __metaclass__ = ABCMeta

    def __init__(self, idx, server_name):
        """
        :param idx: idx of this process
        :param player: An RLEnvironment
        :param server_name: name of the server socket
        """
        super(SimulatorProcess, self).__init__()
        self.idx = int(idx)
        self.server_name = server_name

    def run(self):
        player = self._build_player()
        context = zmq.Context()
        socket = context.socket(zmq.REQ)
        socket.identity = 'simulator-{}'.format(self.idx)
        socket.connect(self.server_name)

        while True:
            state = player.current_state()
            socket.send(dumps(state), copy=False)
            action = loads(socket.recv(copy=False))
            reward, isOver = player.action(action)
            socket.send(dumps((reward, isOver)), copy=False)
            noop = socket.recv(copy=False)

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
        def __init__(self, state, action, reward):
            self.state = state
            self.action = action
            self.reward = reward

    def __init__(self, server_name):
        super(SimulatorMaster, self).__init__()
        self.server_name = server_name
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.ROUTER)
        self.socket.bind(self.server_name)
        self.socket_lock = threading.Lock()
        self.daemon = True

        def clean_context(sok, context):
            sok.close()
            context.term()
        import atexit
        atexit.register(clean_context, self.socket, self.context)

    def run(self):
        self.clients = defaultdict(SimulatorMaster.ClientState)
        while True:
            ident, _, msg = self.socket.recv_multipart()
            #assert  _ == ""
            client = self.clients[ident]
            client.protocol_state = 1 - client.protocol_state   # first flip the state
            if not client.protocol_state == 0:   # state-action
                state = loads(msg)
                self._on_state(state, ident)
            else:       # reward-response
                reward, isOver = loads(msg)
                client.memory[-1].reward = reward
                if isOver:
                    self._on_episode_over(client)
                else:
                    self._on_datapoint(client)
                self.send_multipart_threadsafe([ident, _, dumps('Thanks')])

    def send_multipart_threadsafe(self, data):
        with self.socket_lock:
            self.socket.send_multipart(data)

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

