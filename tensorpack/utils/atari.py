#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: atari.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

from ale_python_interface import ALEInterface
import numpy as np
import time
import os
import cv2
from .utils import get_rng

__all__ = ['AtariDriver']

class AtariDriver(object):
    """
    A driver for atari games.
    """
    def __init__(self, rom_file, frame_skip=1, viz=0):
        """
        :param rom_file: path to the rom
        :param frame_skip: skip every k frames
        :param viz: the delay. visualize the game while running. 0 to disable
        """
        self.ale = ALEInterface()
        self.rng = get_rng(self)

        self.ale.setInt("random_seed", self.rng.randint(999))
        self.ale.setInt("frame_skip", frame_skip)
        self.ale.loadROM(rom_file)
        self.width, self.height = self.ale.getScreenDims()
        self.actions = self.ale.getMinimalActionSet()

        self.viz = viz
        self.romname = os.path.basename(rom_file)
        if self.viz:
            cv2.startWindowThread()
            cv2.namedWindow(self.romname)

        self._reset()
        self.last_image = self._grab_raw_image()

    def _grab_raw_image(self):
        """
        :returns: a 3-channel image
        """
        m = np.zeros(self.height * self.width * 3, dtype=np.uint8)
        self.ale.getScreenRGB(m)
        return m.reshape((self.height, self.width, 3))

    def grab_image(self):
        """
        :returns: a gray-scale image, maximum over the last
        """
        now = self._grab_raw_image()
        ret = np.maximum(now, self.last_image)
        self.last_image = now
        if self.viz:
            cv2.imshow(self.romname, ret)
            time.sleep(self.viz)
        ret = cv2.cvtColor(ret, cv2.COLOR_BGR2YUV)[:,:,0]
        return ret

    def get_num_actions(self):
        """
        :returns: the number of legal actions
        """
        return len(self.actions)

    def _reset(self):
        self.ale.reset_game()

    def next(self, act):
        """
        :param act: an index of the action
        :returns: (next_image, reward, isOver)
        """
        r = self.ale.act(self.actions[act])
        s = self.grab_image()
        isOver = self.ale.game_over()
        if isOver:
            self._reset()
        return (s, r, isOver)

if __name__ == '__main__':
    a = AtariDriver('breakout.bin', viz=True)
    num = a.get_num_actions()
    rng = get_rng(num)
    import time
    while True:
        #im = a.grab_image()
        #cv2.imshow(a.romname, im)
        act = rng.choice(range(num))
        s, r, o = a.next(act)
        time.sleep(0.1)
        print(r, o)

