# -*- coding: utf-8 -*-
# File: graph.py


""" Graph related callbacks"""

import numpy as np
import os

from ..compat import tfv1 as tf
from ..tfutils.common import get_op_tensor_name
from ..utils import logger
from .base import Callback

__all__ = ['RunOp', 'RunUpdateOps', 'ProcessTensors', 'DumpTensors',
           'DumpTensor', 'DumpTensorAsImage', 'DumpParamAsImage', 'CheckNumerics']


class RunOp(Callback):
    """ Run an Op. """

    _chief_only = False

    def __init__(self, op,
                 run_before=True, run_as_trigger=True,
                 run_step=False, verbose=False):
        """
        Args:
            op (tf.Operation or function): an Op, or a function that returns the Op in the graph.
                The function will be called after the main graph has been created (in the :meth:`setup_graph` callback).
            run_before (bool): run the Op before training
            run_as_trigger (bool): run the Op on every :meth:`trigger()` call.
            run_step (bool): run the Op every step (along with training)
            verbose (bool): print logs when the op is run.

        Example:
            The `DQN Example
            <https://github.com/tensorpack/tensorpack/blob/master/examples/DeepQNetwork/>`_
            uses this callback to update target network.
        """
        if not callable(op):
            self.setup_func = lambda: op  # noqa
        else:
            self.setup_func = op
        self.run_before = run_before
        self.run_as_trigger = run_as_trigger
        self.run_step = run_step
        self.verbose = verbose

    def _setup_graph(self):
        self._op = self.setup_func()
        if self.run_step:
            self._fetch = tf.train.SessionRunArgs(fetches=self._op)

    def _before_train(self):
        if self.run_before:
            self._print()
            self._op.run()

    def _trigger(self):
        if self.run_as_trigger:
            self._print()
            self._op.run()

    def _before_run(self, _):
        if self.run_step:
            self._print()
            return self._fetch

    def _print(self):
        if self.verbose:
            logger.info("Running Op {} ...".format(self._op.name))


class RunUpdateOps(RunOp):
    """
    Run ops from the collection UPDATE_OPS every step.
    The ops will be hooked to ``trainer.hooked_sess`` and run along with
    each ``hooked_sess.run`` call.

    Be careful when using ``UPDATE_OPS`` if your model contains more than one sub-networks.
    Perhaps not all updates are supposed to be executed in every iteration.

    This callback is one of the :func:`DEFAULT_CALLBACKS()`.
    """

    def __init__(self, collection=None):
        """
        Args:
            collection (str): collection of ops to run. Defaults to ``tf.GraphKeys.UPDATE_OPS``
        """
        if collection is None:
            collection = tf.GraphKeys.UPDATE_OPS
        name = 'UPDATE_OPS' if collection == tf.GraphKeys.UPDATE_OPS else collection

        def f():
            ops = tf.get_collection(collection)
            if ops:
                logger.info("Applying collection {} of {} ops.".format(name, len(ops)))
                return tf.group(*ops, name='update_ops')
            else:
                return tf.no_op(name='empty_update_ops')

        super(RunUpdateOps, self).__init__(
            f, run_before=False, run_as_trigger=False, run_step=True)


class ProcessTensors(Callback):
    """
    Fetch extra tensors **along with** each training step,
    and call some function over the values.
    It uses ``_{before,after}_run`` method to inject ``tf.train.SessionRunHooks``
    to the session.
    You can use it to print tensors, save tensors to file, etc.

    Example:

    .. code-block:: python

        ProcessTensors(['mycost1', 'mycost2'], lambda c1, c2: print(c1, c2, c1 + c2))
    """
    def __init__(self, names, fn):
        """
        Args:
            names (list[str]): names of tensors
            fn: a function taking all requested tensors as input
        """
        assert isinstance(names, (list, tuple)), names
        self._names = names
        self._fn = fn

    def _setup_graph(self):
        tensors = self.get_tensors_maybe_in_tower(self._names)
        self._fetch = tf.train.SessionRunArgs(fetches=tensors)

    def _before_run(self, _):
        return self._fetch

    def _after_run(self, _, rv):
        results = rv.results
        self._fn(*results)


class DumpTensors(ProcessTensors):
    """
    Dump some tensors to a file.
    Every step this callback fetches tensors and write them to a npz file
    under ``logger.get_logger_dir``.
    The dump can be loaded by ``dict(np.load(filename).items())``.
    """
    def __init__(self, names):
        """
        Args:
            names (list[str]): names of tensors
        """
        assert isinstance(names, (list, tuple)), names
        self._names = names
        dir = logger.get_logger_dir()

        def fn(*args):
            dic = {}
            for name, val in zip(self._names, args):
                dic[name] = val
            fname = os.path.join(
                dir, 'DumpTensor-{}.npz'.format(self.global_step))
            np.savez(fname, **dic)
        super(DumpTensors, self).__init__(names, fn)


class DumpTensorAsImage(Callback):
    """
    Dump a tensor to image(s) to ``logger.get_logger_dir()`` once triggered.

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
        self.log_dir = logger.get_logger_dir()
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


class CheckNumerics(Callback):
    """
    When triggered, check variables in the graph for NaN and Inf.
    Raise exceptions if such an error is found.
    """
    def _setup_graph(self):
        vars = tf.trainable_variables()
        ops = [tf.check_numerics(v, "CheckNumerics['{}']".format(v.op.name)).op for v in vars]
        self._check_op = tf.group(*ops)

    def _trigger(self):
        self._check_op.run()


try:
    import cv2
except ImportError:
    from ..utils.develop import create_dummy_class
    DumpTensorAsImage = create_dummy_class('DumpTensorAsImage', 'cv2')  # noqa

# alias
DumpParamAsImage = DumpTensorAsImage
DumpTensor = DumpTensors
