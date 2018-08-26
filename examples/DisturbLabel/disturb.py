# -*- coding: utf-8 -*-
# File: disturb.py
# Author: Yuxin Wu

from tensorpack.dataflow import ProxyDataFlow, RNGDataFlow


class DisturbLabel(ProxyDataFlow, RNGDataFlow):
    def __init__(self, ds, prob):
        super(DisturbLabel, self).__init__(ds)
        self.prob = prob

    def reset_state(self):
        RNGDataFlow.reset_state(self)
        ProxyDataFlow.reset_state(self)

    def __iter__(self):
        for dp in self.ds:
            img, l = dp
            if self.rng.rand() < self.prob:
                l = self.rng.choice(10)
            yield [img, l]
