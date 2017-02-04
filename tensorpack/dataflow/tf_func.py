#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: tf_func.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import tensorflow as tf
from .base import ProxyDataFlow


""" This file was deprecated """

__all__ = []


class TFFuncMapper(ProxyDataFlow):
    def __init__(self, ds,
                 get_placeholders, symbf, apply_symbf_on_dp, device='/cpu:0'):
        """
        :param get_placeholders: a function returning the placeholders
        :param symbf: a symbolic function taking the placeholders
        :param apply_symbf_on_dp: apply the above function to datapoint
        """
        super(TFFuncMapper, self).__init__(ds)
        self.get_placeholders = get_placeholders
        self.symbf = symbf
        self.apply_symbf_on_dp = apply_symbf_on_dp
        self.device = device

    def reset_state(self):
        super(TFFuncMapper, self).reset_state()
        self.graph = tf.Graph()
        with self.graph.as_default(), \
                tf.device(self.device):
            self.placeholders = self.get_placeholders()
            self.output_vars = self.symbf(self.placeholders)
            self.sess = tf.Session()

            def run_func(vals):
                return self.sess.run(self.output_vars,
                                     feed_dict=dict(zip(self.placeholders, vals)))
            self.run_func = run_func

    def get_data(self):
        for dp in self.ds.get_data():
            dp = self.apply_symbf_on_dp(dp, self.run_func)
            if dp:
                yield dp


if __name__ == '__main__':
    from .raw import FakeData
    ds = FakeData([[224, 224, 3]], 100000, random=False)

    def tf_aug(v):
        v = v[0]
        v = tf.image.random_brightness(v, 0.1)
        v = tf.image.random_contrast(v, 0.8, 1.2)
        v = tf.image.random_flip_left_right(v)
        return v
    ds = TFFuncMapper(ds,
                      lambda: [tf.placeholder(tf.float32, [224, 224, 3], name='img')],
                      tf_aug,
                      lambda dp, f: [f([dp[0]])[0]]
                      )
    # from .prefetch import PrefetchDataZMQ
    # from .image import AugmentImageComponent
    # from . import imgaug
    # ds = AugmentImageComponent(ds,
    #   [imgaug.Brightness(0.1, clip=False),
    #    imgaug.Contrast((0.8, 1.2), clip=False),
    #    imgaug.Flip(horiz=True)
    #   ])
    # ds = PrefetchDataZMQ(ds, 4)
    ds.reset_state()

    import tqdm
    itr = ds.get_data()
    for k in tqdm.trange(100000):
        next(itr)
