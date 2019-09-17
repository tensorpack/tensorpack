# -*- coding: utf-8 -*-
# File: _test.py


import logging
import unittest
import tensorflow as tf
import numpy as np

from .conv2d import Conv2DTranspose
from .pool import FixedUnPooling


class TestModel(unittest.TestCase):

    def run_variable(self, var):
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        if isinstance(var, list):
            return sess.run(var)
        else:
            return sess.run([var])[0]

    def make_variable(self, *args):
        if len(args) > 1:
            return [tf.Variable(k) for k in args]
        else:
            return tf.Variable(args[0])


class TestPool(TestModel):
    def test_FixedUnPooling(self):
        h, w = 3, 4
        scale = 2
        mat = np.random.rand(h, w, 3).astype('float32')
        input = self.make_variable(mat)
        input = tf.reshape(input, [1, h, w, 3])
        output = FixedUnPooling('unpool', input, scale)
        res = self.run_variable(output)
        self.assertEqual(res.shape, (1, scale * h, scale * w, 3))

        # mat is on corner
        ele = res[0, ::scale, ::scale, 0]
        self.assertTrue((ele == mat[:, :, 0]).all())
        # the rest are zeros
        res[0, ::scale, ::scale, :] = 0
        self.assertTrue((res == 0).all())

# Below was originally for the BilinearUpsample layer used in the HED example
#     def test_BilinearUpSample(self):
#         h, w = 12, 12
#         scale = 2
#
#         mat = np.random.rand(h, w).astype('float32')
#         inp = self.make_variable(mat)
#         inp = tf.reshape(inp, [1, h, w, 1])
#
#         output = BilinearUpSample(inp, scale)
#         res = self.run_variable(output)[0, :, :, 0]
#
#         from skimage.transform import rescale
#         res2 = rescale(mat, scale, mode='edge')
#
#         diff = np.abs(res2 - res)
#
#         # if not diff.max() < 1e-4:
#         #     import IPython
#         #     IPython.embed(config=IPython.terminal.ipapp.load_default_config())
#         self.assertTrue(diff.max() < 1e-4, diff.max())


class TestConv2DTranspose(TestModel):
    def setUp(self):
        tf.reset_default_graph()

    def test_shape_match(self):
        h, w = 12, 18
        input = self.make_variable(np.random.rand(1, h, w, 3).astype("float32"))
        for padding in ["same"]:
            for stride in [1, 2]:
                output = Conv2DTranspose(
                    'deconv_s{}_pad{}'.format(stride, padding),
                    input, 20, 3, strides=stride, padding=padding)

                static_shape = output.shape
                dynamic_shape = self.run_variable(output).shape
                self.assertTrue(static_shape == dynamic_shape)


def run_test_case(case):
    suite = unittest.TestLoader().loadTestsFromTestCase(case)
    unittest.TextTestRunner(verbosity=2).run(suite)


if __name__ == '__main__':
    from tensorpack.utils import logger
    logger.setLevel(logging.CRITICAL)
    unittest.main()
