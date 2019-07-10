#-*- coding: utf-8 -*-

import six
import unittest
import tensorflow as tf

from ..utils import logger
from .scope_utils import under_name_scope


class ScopeUtilsTest(unittest.TestCase):

    @under_name_scope(name_scope='s')
    def _f(self, check=True):
        if check:
            assert tf.get_default_graph().get_name_scope().endswith('s')
        return True

    def test_under_name_scope(self):
        self.assertTrue(self._f())
        with self.assertRaises(AssertionError):
            self._f()  # name conflict

    @unittest.skipIf(six.PY2, "assertLogs not supported in Python 2")
    def test_under_name_scope_warning(self):
        x = tf.placeholder(tf.float32, [3])
        tf.nn.relu(x, name='s')
        with self.assertLogs(logger=logger._logger, level='WARNING'):
            self._f(check=False, name_scope='s')


if __name__ == '__main__':
    unittest.main()
