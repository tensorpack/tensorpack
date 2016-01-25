#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: image_sample.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import tensorflow as tf

from ._common import layer_register

__all__ = ['ImageSample']

# XXX TODO ugly.
# really need to fix this after tensorflow supports multiple indexing
# See github:tensorflow#418,#206
def sample(img, coords):
    """
    img: bxhxwxc
    coords: bxh2xw2x2 (y, x) integer
    """
    shape = img.get_shape().as_list()[1:]
    shape2 = coords.get_shape().as_list()[1:3]
    max_coor = tf.constant([shape[0] - 1, shape[1] - 1])
    coords = tf.minimum(coords, max_coor)
    coords = tf.maximum(coords, tf.constant(0))

    w = shape[1]
    coords = tf.reshape(coords, [-1, 2])
    coords = tf.matmul(coords, tf.constant([[w], [1]]))
    coords = tf.reshape(coords, [-1] + shape2)
    # bxh2xw2

    batch_add = tf.range(tf.shape(img)[0]) * (shape[0] * shape[1])
    batch_add = tf.reshape(batch_add, [-1, 1, 1])   #bx1x1

    flat_coords = coords + batch_add

    img = tf.reshape(img, [-1, shape[2]])   #bhw x c
    sampled = tf.gather(img, flat_coords)
    return sampled

@layer_register()
def ImageSample(template, mapping, interpolate):
    """
    Sample an image from template, using the given coordinate
    template: bxhxwxc
    mapping: bxh2xw2x2  (y, x) real-value coordinates
    interpolate: 'nearest'
    Return: bxh2xw2xc
    """

    if interpolate == 'nearest':
        mapping = tf.cast(tf.floor(mapping + 0.5), tf.int32)
        return sample(template, mapping)
    else:
        raise NotImplementedError()

from _test import TestModel
class TestSample(TestModel):
    def test_sample(self):
        import numpy as np
        h, w = 3, 4
        def np_sample(img, coords):
            # a reference implementation
            coords = np.maximum(coords, 0)
            coords = np.minimum(coords,
                                np.array([img.shape[1]-1, img.shape[2]-1]))
            xs = coords[:,:,:,1].reshape((img.shape[0], -1))
            ys = coords[:,:,:,0].reshape((img.shape[0], -1))

            ret = np.zeros((img.shape[0], coords.shape[1], coords.shape[2],
                            img.shape[3]), dtype='float32')
            for k in range(img.shape[0]):
                xss, yss = xs[k], ys[k]
                ret[k,:,:,:] = img[k,yss,xss,:].reshape((coords.shape[1],
                                                         coords.shape[2], 3))
            return ret

        bimg = np.random.rand(2, h, w, 3).astype('float32')

        #mat = np.array([
            #[[[1,1], [1.2,1.2]], [[-1, -1], [2.5, 2.5]]],
            #[[[1,1], [1.2,1.2]], [[-1, -1], [2.5, 2.5]]]
        #], dtype='float32')  #2x2x2x2
        mat = (np.random.rand(2, 5, 5, 2) - 0.2) * np.array([h + 3, w + 3])
        true_res = np_sample(bimg, np.floor(mat + 0.5).astype('int32'))

        inp, mapping = self.make_variable(bimg, mat)
        output = ImageSample('sample', inp, mapping, 'nearest')
        res = self.run_variable(output)

        self.assertTrue((res == true_res).all())
