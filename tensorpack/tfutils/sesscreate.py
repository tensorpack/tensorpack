# -*- coding: utf-8 -*-
# File: sesscreate.py


from ..compat import tfv1 as tf
from ..utils import logger
from .common import get_default_sess_config

__all__ = ['NewSessionCreator', 'ReuseSessionCreator', 'SessionCreatorAdapter']

"""
A SessionCreator should:
    create the session
    initialize all variables
    return a session that is ready to use
    not finalize the graph
"""


_WRN1 = """User-provided custom session config may not work due to TF bugs. If you saw logs like
```
tensorflow/core/common_runtime/gpu/gpu_device.cc:1433] Found device 0 with properties:
```
before this line, then your GPU has been initialized and custom GPU options may not take effect. """

_WRN2 = """To workaround this issue, you can do one of the following:
1. Avoid initializing the GPU too early. Find code that initializes the GPU and skip it.
   Typically examples are: creating a session; check GPU availability; check GPU number.
2. Manually set your GPU options earlier. You can create a session with custom
   GPU options at the beginning of your program, as described in
   https://github.com/tensorpack/tensorpack/issues/497
"""


class NewSessionCreator(tf.train.SessionCreator):
    def __init__(self, target='', config=None):
        """
        Args:
            target, config: same as :meth:`Session.__init__()`.
            config: a :class:`tf.ConfigProto` instance, defaults to :func:`tfutils.get_default_sess_config()`
        """
        self.target = target

        if config is None:
            # distributed trainer doesn't support user-provided config
            # we set this attribute so that they can check
            self.user_provided_config = False
            config = get_default_sess_config()
        else:
            self.user_provided_config = True
            logger.warn(_WRN1)
            logger.warn(_WRN2)
        self.config = config

    def create_session(self):
        sess = tf.Session(target=self.target, config=self.config)

        def blocking_op(x):
            """
            Whether an op is possibly blocking.
            """
            if x.op_def is not None and not x.op_def.is_stateful:
                return False
            if "Dequeue" in x.type or "Enqueue" in x.type:
                return True
            if "Unstage" in x.type:
                return True
            if x.type in ["ZMQPull"]:
                return True
            return False

        def run(op):
            try:
                from tensorflow.contrib.graph_editor import get_backward_walk_ops  # deprecated
            except ImportError:
                from tensorflow.python.ops.op_selector import get_backward_walk_ops

            deps = get_backward_walk_ops(op, control_inputs=True)
            for dep_op in deps:
                if blocking_op(dep_op):
                    logger.warn(
                        "Initializer '{}' depends on a blocking op '{}'. "
                        "This initializer is likely to hang!".format(
                            op.name, dep_op.name))

            sess.run(op)

        run(tf.global_variables_initializer())
        run(tf.local_variables_initializer())
        run(tf.tables_initializer())
        return sess


class ReuseSessionCreator(tf.train.SessionCreator):
    """
    Returns an existing session.
    """
    def __init__(self, sess):
        """
        Args:
            sess (tf.Session): the session to reuse
        """
        self.sess = sess

    def create_session(self):
        return self.sess


class SessionCreatorAdapter(tf.train.SessionCreator):
    """
    Apply a function on the output of a SessionCreator. Can be used to create a debug session.

    Note:
    Since TF 1.6, debug session may not work properly with Monitored session.
    This is a tensorflow bug. To use tfdbg, use the :class:`TFLocalCLIDebugHook` callback instead.
    """
    def __init__(self, session_creator, func):
        """
        Args:
            session_creator (tf.train.SessionCreator): a session creator
            func (tf.Session -> tf.Session): takes a session created by
            ``session_creator``, and return a new session to be returned by ``self.create_session``
        """
        self._creator = session_creator
        self._func = func

    def create_session(self):
        sess = self._creator.create_session()
        return self._func(sess)
