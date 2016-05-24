#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: atari.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import numpy as np
import time
import os
import cv2
from collections import deque
from ...utils import get_rng, logger
from ..RL import RLEnvironment

try:
    from ale_python_interface import ALEInterface
except ImportError:
    logger.warn("Cannot import ale_python_interface, Atari won't be available.")

__all__ = ['AtariDriver', 'AtariPlayer']

class AtariDriver(object):
    """
    A wrapper for atari emulator.
    """
    def __init__(self, rom_file,
            frame_skip=1, viz=0, height_range=(None,None)):
        """
        :param rom_file: path to the rom
        :param frame_skip: skip every k frames
        :param viz: the delay. visualize the game while running. 0 to disable
        """
        self.ale = ALEInterface()
        self.rng = get_rng(self)

        self.ale.setInt("random_seed", self.rng.randint(self.rng.randint(0, 1000)))
        self.ale.setInt("frame_skip", frame_skip)
        self.ale.loadROM(rom_file)
        self.width, self.height = self.ale.getScreenDims()
        self.actions = self.ale.getMinimalActionSet()

        if isinstance(viz, int):
            viz = float(viz)
        self.viz = viz
        self.romname = os.path.basename(rom_file)
        if self.viz and isinstance(self.viz, float):
            cv2.startWindowThread()
            cv2.namedWindow(self.romname)

        self._reset()
        self.last_image = self._grab_raw_image()
        self.framenum = 0
        self.height_range = height_range

    def _grab_raw_image(self):
        """
        :returns: the current 3-channel image
        """
        m = np.zeros(self.height * self.width * 3, dtype=np.uint8)
        self.ale.getScreenRGB(m)
        return m.reshape((self.height, self.width, 3))

    def grab_image(self):
        """
        :returns: a gray-scale image, max-pooled over the last frame.
        """
        now = self._grab_raw_image()
        ret = np.maximum(now, self.last_image)
        self.last_image = now
        ret = ret[self.height_range[0]:self.height_range[1],:]   # several online repos all use this
        if self.viz:
            if isinstance(self.viz, float):
                cv2.imshow(self.romname, ret)
                time.sleep(self.viz)
            else:
                cv2.imwrite("{}/{:06d}.jpg".format(self.viz, self.framenum), ret)
                self.framenum += 1
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

class AtariPlayer(RLEnvironment):
    """ An Atari game player with limited memory and FPS"""
    def __init__(self, driver, hist_len=4, action_repeat=4, image_shape=(84,84)):
        """
        :param driver: an `AtariDriver` instance.
        :param hist_len: history(memory) length
        :param action_repeat: repeat each action `action_repeat` times and skip those frames
        :param image_shape: the shape of the observed image
        """
        super(AtariPlayer, self).__init__()
        for k, v in locals().items():
            if k != 'self':
                setattr(self, k, v)
        self.last_act = 0
        self.frames = deque(maxlen=hist_len)
        self.current_accum_score = 0
        self.restart()

    def restart(self):
        """
        Restart the game and populate frames with the beginning frame
        """
        self.current_accum_score = 0
        self.frames.clear()
        s = self.driver.grab_image()

        s = cv2.resize(s, self.image_shape)
        for _ in range(self.hist_len):
            self.frames.append(s)

    def current_state(self):
        """
        Return a current state of shape `image_shape + (hist_len,)`
        """
        return self._build_state()

    def action(self, act):
        """
        Perform an action
        :param act: index of the action
        :returns: (new_frame, reward, isOver)
        """
        self.last_act = act
        return self._observe()

    def _build_state(self):
        assert len(self.frames) == self.hist_len
        m = np.array(self.frames)
        m = m.transpose([1,2,0])
        return m

    def _observe(self):
        """ if isOver==True, current_state will return the new episode
        """
        totr = 0
        for k in range(self.action_repeat):
            s, r, isOver = self.driver.next(self.last_act)
            totr += r
            if isOver:
                break
        s = cv2.resize(s, self.image_shape)
        self.current_accum_score += totr
        self.frames.append(s)
        if isOver:
            self.stats['score'].append(self.current_accum_score)
            self.restart()
        return (totr, isOver)

    def get_stat(self):
        try:
            return {'avg_score': np.mean(self.stats['score']),
                    'max_score': float(np.max(self.stats['score'])) }
        except ValueError:
            return {}

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

