#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: dump.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import os
import scipy.misc
from scipy.misc import imsave

from .base import Callback
from ..utils import logger

__all__ = ['DumpParamAsImage']

class DumpParamAsImage(Callback):
    def __init__(self, var_name, prefix=None, map_func=None, scale=255):
        """
        map_func: map the value of the variable to an image or list of images, default to identity
            images should have shape [h, w] or [h, w, c].
        scale: a multiplier on pixel values, applied after map_func. default to 255
        """
        self.var_name = var_name
        self.func = map_func
        if prefix is None:
            self.prefix = self.var_name
        else:
            self.prefix = prefix
        self.log_dir = logger.LOG_DIR
        self.scale = scale

    def _before_train(self):
        self.var = self.graph.get_tensor_by_name(self.var_name)

    def _trigger_epoch(self):
        val = self.sess.run(self.var)
        if self.func is not None:
            val = self.func(val)
        if isinstance(val, list):
            for idx, im in enumerate(val):
                self._dump_image(im, idx)
        else:
            self._dump_image(val)

    def _dump_image(self, im, idx=None):
        assert im.ndim in [2, 3], str(im.ndim)
        fname = os.path.join(
            self.log_dir,
            self.prefix + '-ep{:03d}{}.png'.format(
                self.epoch_num, '-' + str(idx) if idx else ''))
        imsave(fname, (im * self.scale).astype('uint8'))

