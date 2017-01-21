#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: embedding_data.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import numpy as np
from tensorpack.dataflow import dataset, BatchData


def get_test_data():
    ds = dataset.Mnist('test')
    ds = BatchData(ds, 128)
    return ds


class MnistPairs(dataset.Mnist):
    """We could also write

    .. code::

        ds = dataset.Mnist('train')
        ds = JoinData([ds, ds])
        ds = MapData(ds, lambda dp: [dp[0], dp[2], dp[1] == dp[3]])
        ds = BatchData(ds, 128 // 2)

    but then the positives pairs would be really rare (p=0.1).
    """
    def __init__(self, train_or_test):
        super(MnistPairs, self).__init__(train_or_test, shuffle=False)
        # now categorize these digits
        self.data_dict = []
        for clazz in range(0, 10):
            clazz_filter = np.where(self.labels == clazz)
            self.data_dict.append(self.images[clazz_filter])

    def get_data(self):
        while True:
            pick_label = self.rng.randint(10)
            pick_other = pick_label
            y = self.rng.randint(2)

            if y == 0:
                # pair with different digits
                offset = self.rng.randint(9)
                pick_other = (pick_label + offset + 1) % 10
                assert not pick_label == pick_other

            l = self.rng.randint(len(self.data_dict[pick_label]))
            r = self.rng.randint(len(self.data_dict[pick_other]))

            l = np.reshape(self.data_dict[pick_label][l], [28, 28]).astype(np.float32)
            r = np.reshape(self.data_dict[pick_other][r], [28, 28]).astype(np.float32)

            yield [l, r, y]


class MnistTriplets(dataset.Mnist):
    def __init__(self, train_or_test):
        super(MnistTriplets, self).__init__(train_or_test, shuffle=False)

        # now categorize these digits
        self.data_dict = []
        for clazz in range(0, 10):
            clazz_filter = np.where(self.labels == clazz)
            self.data_dict.append(self.images[clazz_filter])

    def get_data(self):
        while True:
            pick_label = self.rng.randint(10)
            offset = self.rng.randint(9)
            pick_other = (pick_label + offset + 1) % 10
            assert not pick_label == pick_other

            a = self.rng.randint(len(self.data_dict[pick_label]))
            p = self.rng.randint(len(self.data_dict[pick_label]))
            n = self.rng.randint(len(self.data_dict[pick_other]))

            a = np.reshape(self.data_dict[pick_label][a], [28, 28]).astype(np.float32)
            p = np.reshape(self.data_dict[pick_label][p], [28, 28]).astype(np.float32)
            n = np.reshape(self.data_dict[pick_other][n], [28, 28]).astype(np.float32)

            yield [a, p, n]
