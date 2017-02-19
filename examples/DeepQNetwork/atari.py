#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: atari.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import numpy as np
import time
import os
import cv2
from collections import deque
import threading
import six
from six.moves import range
from tensorpack.utils import (get_rng, logger, execute_only_once)
from tensorpack.utils.fs import get_dataset_path
from tensorpack.utils.stats import StatCounter

from tensorpack.RL.envbase import RLEnvironment, DiscreteActionSpace

from ale_python_interface import ALEInterface

__all__ = ['AtariPlayer']

ROM_URL = "https://github.com/openai/atari-py/tree/master/atari_py/atari_roms"
_ALE_LOCK = threading.Lock()


class AtariPlayer(RLEnvironment):
    """
    A wrapper for atari emulator.
    Will automatically restart when a real episode ends (isOver might be just
    lost of lives but not game over).
    """

    def __init__(self, rom_file, viz=0, height_range=(None, None),
                 frame_skip=4, image_shape=(84, 84), nullop_start=30,
                 live_lost_as_eoe=True):
        """
        :param rom_file: path to the rom
        :param frame_skip: skip every k frames and repeat the action
        :param image_shape: (w, h)
        :param height_range: (h1, h2) to cut
        :param viz: visualization to be done.
            Set to 0 to disable.
            Set to a positive number to be the delay between frames to show.
            Set to a string to be a directory to store frames.
        :param nullop_start: start with random number of null ops
        :param live_losts_as_eoe: consider lost of lives as end of episode.  useful for training.
        """
        super(AtariPlayer, self).__init__()
        if not os.path.isfile(rom_file) and '/' not in rom_file:
            rom_file = get_dataset_path('atari_rom', rom_file)
        assert os.path.isfile(rom_file), \
            "rom {} not found. Please download at {}".format(rom_file, ROM_URL)

        try:
            ALEInterface.setLoggerMode(ALEInterface.Logger.Warning)
        except AttributeError:
            if execute_only_once():
                logger.warn("You're not using latest ALE")

        # avoid simulator bugs: https://github.com/mgbellemare/Arcade-Learning-Environment/issues/86
        with _ALE_LOCK:
            self.ale = ALEInterface()
            self.rng = get_rng(self)
            self.ale.setInt(b"random_seed", self.rng.randint(0, 30000))
            self.ale.setBool(b"showinfo", False)

            self.ale.setInt(b"frame_skip", 1)
            self.ale.setBool(b'color_averaging', False)
            # manual.pdf suggests otherwise.
            self.ale.setFloat(b'repeat_action_probability', 0.0)

            # viz setup
            if isinstance(viz, six.string_types):
                assert os.path.isdir(viz), viz
                self.ale.setString(b'record_screen_dir', viz)
                viz = 0
            if isinstance(viz, int):
                viz = float(viz)
            self.viz = viz
            if self.viz and isinstance(self.viz, float):
                self.windowname = os.path.basename(rom_file)
                cv2.startWindowThread()
                cv2.namedWindow(self.windowname)

            self.ale.loadROM(rom_file.encode('utf-8'))
        self.width, self.height = self.ale.getScreenDims()
        self.actions = self.ale.getMinimalActionSet()

        self.live_lost_as_eoe = live_lost_as_eoe
        self.frame_skip = frame_skip
        self.nullop_start = nullop_start
        self.height_range = height_range
        self.image_shape = image_shape

        self.current_episode_score = StatCounter()
        self.restart_episode()

    def _grab_raw_image(self):
        """
        :returns: the current 3-channel image
        """
        m = self.ale.getScreenRGB()
        return m.reshape((self.height, self.width, 3))

    def current_state(self):
        """
        :returns: a gray-scale (h, w) uint8 image
        """
        ret = self._grab_raw_image()
        # max-pooled over the last screen
        ret = np.maximum(ret, self.last_raw_screen)
        if self.viz:
            if isinstance(self.viz, float):
                cv2.imshow(self.windowname, ret)
                time.sleep(self.viz)
        ret = ret[self.height_range[0]:self.height_range[1], :].astype('float32')
        # 0.299,0.587.0.114. same as rgb2y in torch/image
        ret = cv2.cvtColor(ret, cv2.COLOR_RGB2GRAY)
        ret = cv2.resize(ret, self.image_shape)
        return ret.astype('uint8')  # to save some memory

    def get_action_space(self):
        return DiscreteActionSpace(len(self.actions))

    def finish_episode(self):
        self.stats['score'].append(self.current_episode_score.sum)

    def restart_episode(self):
        self.current_episode_score.reset()
        with _ALE_LOCK:
            self.ale.reset_game()

        # random null-ops start
        n = self.rng.randint(self.nullop_start)
        self.last_raw_screen = self._grab_raw_image()
        for k in range(n):
            if k == n - 1:
                self.last_raw_screen = self._grab_raw_image()
            self.ale.act(0)

    def action(self, act):
        """
        :param act: an index of the action
        :returns: (reward, isOver)
        """
        oldlives = self.ale.lives()
        r = 0
        for k in range(self.frame_skip):
            if k == self.frame_skip - 1:
                self.last_raw_screen = self._grab_raw_image()
            r += self.ale.act(self.actions[act])
            newlives = self.ale.lives()
            if self.ale.game_over() or \
                    (self.live_lost_as_eoe and newlives < oldlives):
                break

        self.current_episode_score.feed(r)
        isOver = self.ale.game_over()
        if self.live_lost_as_eoe:
            isOver = isOver or newlives < oldlives
        if isOver:
            self.finish_episode()
        if self.ale.game_over():
            self.restart_episode()
        return (r, isOver)


if __name__ == '__main__':
    import sys

    def benchmark():
        a = AtariPlayer(sys.argv[1], viz=False, height_range=(28, -8))
        num = a.get_action_space().num_actions()
        rng = get_rng(num)
        start = time.time()
        cnt = 0
        while True:
            act = rng.choice(range(num))
            r, o = a.action(act)
            a.current_state()
            cnt += 1
            if cnt == 5000:
                break
        print(time.time() - start)

    if len(sys.argv) == 3 and sys.argv[2] == 'benchmark':
        import threading
        import multiprocessing
        for k in range(3):
            # th = multiprocessing.Process(target=benchmark)
            th = threading.Thread(target=benchmark)
            th.start()
            time.sleep(0.02)
        benchmark()
    else:
        a = AtariPlayer(sys.argv[1],
                        viz=0.03, height_range=(28, -8))
        num = a.get_action_space().num_actions()
        rng = get_rng(num)
        import time
        while True:
            # im = a.grab_image()
            # cv2.imshow(a.romname, im)
            act = rng.choice(range(num))
            print(act)
            r, o = a.action(act)
            a.current_state()
            # time.sleep(0.1)
            print(r, o)
