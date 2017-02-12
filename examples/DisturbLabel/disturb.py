#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: disturb.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

from tensorpack import ProxyDataFlow, get_rng


class DisturbLabel(ProxyDataFlow):
    def __init__(self, ds, prob):
        super(DisturbLabel, self).__init__(ds)
        self.prob = prob

    def reset_state(self):
        super(DisturbLabel, self).reset_state()
        self.rng = get_rng(self)

    def get_data(self):
        for dp in self.ds.get_data():
            img, l = dp
            if self.rng.rand() < self.prob:
                l = self.rng.choice(10)
            yield [img, l]
