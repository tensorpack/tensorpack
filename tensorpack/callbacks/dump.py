# -*- coding: UTF-8 -*-
# File: dump.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import os
import cv2
import numpy as np

from .base import Callback
from ..utils import logger
from ..tfutils import get_op_tensor_name

__all__ = ['DumpParamAsImage']


class DumpParamAsImage(Callback):
    """
    Dump a variable to image(s) to ``logger.LOG_DIR`` after every epoch.
    """

    def __init__(self, var_name, prefix=None, map_func=None, scale=255, clip=False):
        """
        Args:
            var_name (str): the name of the variable.
            prefix (str): the filename prefix for saved images. Defaults to the Op name.
            map_func: map the value of the variable to an image or list of
                 images of shape [h, w] or [h, w, c]. If None, will use identity.
            scale (float): a multiplier on pixel values, applied after map_func.
            clip (bool): whether to clip the result to [0, 255].
        """
        op_name, self.var_name = get_op_tensor_name(var_name)
        self.func = map_func
        if prefix is None:
            self.prefix = op_name
        else:
            self.prefix = prefix
        self.log_dir = logger.LOG_DIR
        self.scale = scale
        self.clip = clip

    def _before_train(self):
        # TODO might not work for multiGPU?
        self.var = self.graph.get_tensor_by_name(self.var_name)

    def _trigger_epoch(self):
        val = self.trainer.sess.run(self.var)
        if self.func is not None:
            val = self.func(val)
        if isinstance(val, list) or val.ndim == 4:
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
        res = im * self.scale
        if self.clip:
            res = np.clip(res, 0, 255)
        cv2.imwrite(fname, res.astype('uint8'))
