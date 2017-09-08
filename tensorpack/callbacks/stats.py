# -*- coding: utf-8 -*-
# File: stats.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import os
import tensorflow as tf
import numpy as np
from six.moves import zip

from .base import Callback
from ..utils import logger
from ..tfutils.common import get_op_tensor_name, get_tensors_by_names

__all__ = ['SendStat', 'DumpParamAsImage', 'InjectShell', 'DumpTensor']


class SendStat(Callback):
    """ An equivalent of :class:`SendMonitorData`, but as a normal callback. """
    def __init__(self, command, names):
        self.command = command
        if not isinstance(names, list):
            names = [names]
        self.names = names

    def _trigger(self):
        M = self.trainer.monitors
        v = {k: M.get_latest(k) for k in self.names}
        cmd = self.command.format(**v)
        ret = os.system(cmd)
        if ret != 0:
            logger.error("Command {} failed with ret={}!".format(cmd, ret))


class InjectShell(Callback):
    """
    Allow users to create a specific file as a signal to pause
    and iteratively debug the training.
    When triggered, it detects whether the file exists, and opens an
    IPython/pdb shell if yes.
    In the shell, `self` is this callback, `self.trainer` is the trainer, and
    from that you can access everything else.
    """

    def __init__(self, file='INJECT_SHELL.tmp', shell='ipython'):
        """
        Args:
           file (str): if this file exists, will open a shell.
           shell (str): one of 'ipython', 'pdb'
        """
        self._file = file
        assert shell in ['ipython', 'pdb']
        self._shell = shell
        logger.info("Create a file '{}' to open {} shell.".format(file, shell))

    def _trigger(self):
        if os.path.isfile(self._file):
            logger.info("File {} exists, entering shell.".format(self._file))
            self._inject()

    def _inject(self):
        trainer = self.trainer   # noqa
        if self._shell == 'ipython':
            import IPython as IP    # noqa
            IP.embed()
        elif self._shell == 'pdb':
            import pdb   # noqa
            pdb.set_trace()

    def _after_train(self):
        if os.path.isfile(self._file):
            os.unlink(self._file)


class DumpParamAsImage(Callback):
    """
    Dump a tensor to image(s) to ``logger.LOG_DIR`` after every epoch.

    Note that it requires the tensor is directly evaluable, i.e. either inputs
    are not its dependency (e.g. the weights of the model), or the inputs are
    feedfree (in which case this callback will take an extra datapoint from the input pipeline).
    """

    def __init__(self, tensor_name, prefix=None, map_func=None, scale=255):
        """
        Args:
            tensor_name (str): the name of the tensor.
            prefix (str): the filename prefix for saved images. Defaults to the Op name.
            map_func: map the value of the tensor to an image or list of
                 images of shape [h, w] or [h, w, c]. If None, will use identity.
            scale (float): a multiplier on pixel values, applied after map_func.
        """
        op_name, self.tensor_name = get_op_tensor_name(tensor_name)
        self.func = map_func
        if prefix is None:
            self.prefix = op_name
        else:
            self.prefix = prefix
        self.log_dir = logger.LOG_DIR
        self.scale = scale

    def _before_train(self):
        self._tensor = self.graph.get_tensor_by_name(self.tensor_name)

    def _trigger(self):
        val = self.trainer.sess.run(self._tensor)
        if self.func is not None:
            val = self.func(val)
        if isinstance(val, list) or val.ndim == 4:
            for idx, im in enumerate(val):
                self._dump_image(im, idx)
        else:
            self._dump_image(val)
        self.trainer.monitors.put_image(self.prefix, val)

    def _dump_image(self, im, idx=None):
        assert im.ndim in [2, 3], str(im.ndim)
        fname = os.path.join(
            self.log_dir,
            self.prefix + '-ep{:03d}{}.png'.format(
                self.epoch_num, '-' + str(idx) if idx else ''))
        res = im * self.scale
        res = np.clip(res, 0, 255)
        cv2.imwrite(fname, res.astype('uint8'))


class DumpTensor(Callback):
    """
    Dump some tensors to a file.
    Every step this callback fetches tensors and write them to a npz file under ``logger.LOG_DIR``.
    The dump can be loaded by ``dict(np.load(filename).items())``.
    """
    # TODO run as trigger
    def __init__(self, names):
        """
        Args:
            names (list[str]): names of tensors
        """
        assert isinstance(names, (list, tuple)), names
        self._names = names
        self._dir = logger.LOG_DIR

    def _setup_graph(self):
        tensors = get_tensors_by_names(self._names)
        self._fetch = tf.train.SessionRunArgs(fetches=tensors)

    def _before_run(self, _):
        return self._fetch

    def _after_run(self, _, rv):
        results = rv.results
        dic = {}
        for name, val in zip(self._names, results):
            dic[name] = val
        fname = os.path.join(
            self._dir, 'DumpTensor-{}.npz'.format(self.global_step))
        np.savez(fname, **dic)


try:
    import cv2
except ImportError:
    from ..utils.develop import create_dummy_class
    DumpParamAsImage = create_dummy_class('DumpParamAsImage', 'cv2')  # noqa
