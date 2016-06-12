#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: atari.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import numpy as np
import time, os
import cv2
from collections import deque
import threading
import six
from six.moves import range
from ..utils import get_rng, logger, memoized, get_dataset_dir
from ..utils.stat import StatCounter

from .envbase import RLEnvironment, DiscreteActionSpace

try:
    from ale_python_interface import ALEInterface
except ImportError:
    logger.warn("Cannot import ale_python_interface, Atari won't be available.")

__all__ = ['AtariPlayer']

@memoized
def log_once():
    logger.warn("https://github.com/mgbellemare/Arcade-Learning-Environment/pull/171 is not merged!")

_ALE_LOCK = threading.Lock()

class AtariPlayer(RLEnvironment):
    """
    A wrapper for atari emulator.
    NOTE: will automatically restart when a real episode ends
    """
    def __init__(self, rom_file, viz=0, height_range=(None,None),
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
            rom_file = os.path.join(get_dataset_dir('atari_rom'), rom_file)
        assert os.path.isfile(rom_file), "rom {} not found".format(rom_file)

        try:
            ALEInterface.setLoggerMode(ALEInterface.Logger.Warning)
        except AttributeError:
            log_once()

        # avoid simulator bugs: https://github.com/mgbellemare/Arcade-Learning-Environment/issues/86
        with _ALE_LOCK:
            self.ale = ALEInterface()
            self.rng = get_rng(self)

            self.ale.setInt("random_seed", self.rng.randint(0, 10000))
            self.ale.setBool("showinfo", False)

            self.ale.setInt("frame_skip", 1)
            self.ale.setBool('color_averaging', False)
            # manual.pdf suggests otherwise.
            self.ale.setFloat('repeat_action_probability', 0.0)

            # viz setup
            if isinstance(viz, six.string_types):
                assert os.path.isdir(viz), viz
                self.ale.setString('record_screen_dir', viz)
                viz = 0
            if isinstance(viz, int):
                viz = float(viz)
            self.viz = viz
            if self.viz and isinstance(self.viz, float):
                self.windowname = os.path.basename(rom_file)
                cv2.startWindowThread()
                cv2.namedWindow(self.windowname)

            self.ale.loadROM(rom_file)
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
        :returns: a gray-scale (h, w, 1) float32 image
        """
        ret = self._grab_raw_image()
        # max-pooled over the last screen
        ret = np.maximum(ret, self.last_raw_screen)
        if self.viz:
            if isinstance(self.viz, float):
                #m = cv2.resize(ret, (1920,1200))
                cv2.imshow(self.windowname, ret)
                time.sleep(self.viz)
        ret = ret[self.height_range[0]:self.height_range[1],:].astype('float32')
        # 0.299,0.587.0.114. same as rgb2y in torch/image
        ret = cv2.cvtColor(ret, cv2.COLOR_RGB2GRAY)
        ret = cv2.resize(ret, self.image_shape)
        ret = np.expand_dims(ret, axis=2)
        return ret

    def get_action_space(self):
        return DiscreteActionSpace(len(self.actions))

    def restart_episode(self):
        if self.current_episode_score.count > 0:
            self.stats['score'].append(self.current_episode_score.sum)
        self.current_episode_score.reset()
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
        if isOver:
            self.restart_episode()
        if self.live_lost_as_eoe:
            isOver = isOver or newlives < oldlives
        return (r, isOver)

if __name__ == '__main__':
    import sys
    import time

    def benchmark():
        a = AtariPlayer(sys.argv[1], viz=False, height_range=(28,-8))
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
        print time.time() - start

    if len(sys.argv) == 3 and sys.argv[2] == 'benchmark':
        import threading, multiprocessing
        for k in range(3):
            #th = multiprocessing.Process(target=benchmark)
            th = threading.Thread(target=benchmark)
            th.start()
            time.sleep(0.02)
        benchmark()
    else:
        a = AtariPlayer(sys.argv[1],
                viz=0.03, height_range=(28,-8))
        num = a.get_action_space().num_actions()
        rng = get_rng(num)
        import time
        while True:
            #im = a.grab_image()
            #cv2.imshow(a.romname, im)
            act = rng.choice(range(num))
            print(act)
            r, o = a.action(act)
            a.current_state()
            #time.sleep(0.1)
            print(r, o)

