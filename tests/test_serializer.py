#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import os
import unittest

from tensorpack.dataflow import HDF5Serializer, LMDBSerializer, NumpySerializer, TFRecordSerializer
from tensorpack.dataflow.base import DataFlow


def delete_file_if_exists(fn):
    try:
        os.remove(fn)
    except OSError:
        pass


class SeededFakeDataFlow(DataFlow):
    """docstring for SeededFakeDataFlow"""

    def __init__(self, seed=42, size=32):
        super(SeededFakeDataFlow, self).__init__()
        self.seed = seed
        self._size = size
        self.cache = []

    def reset_state(self):
        np.random.seed(self.seed)
        for _ in range(self._size):
            label = np.random.randint(low=0, high=10)
            img = np.random.randn(28, 28, 3)
            self.cache.append([label, img])

    def __len__(self):
        return self._size

    def __iter__(self):
        for dp in self.cache:
            yield dp


class SerializerTest(unittest.TestCase):

    def run_write_read_test(self, file, serializer, w_args, w_kwargs, r_args, r_kwargs, error_msg):
        try:
            delete_file_if_exists(file)

            ds_expected = SeededFakeDataFlow()
            serializer.save(ds_expected, file, *w_args, **w_kwargs)
            ds_actual = serializer.load(file, *r_args, **r_kwargs)

            ds_actual.reset_state()
            ds_expected.reset_state()

            for dp_expected, dp_actual in zip(ds_expected.__iter__(), ds_actual.__iter__()):
                self.assertEqual(dp_expected[0], dp_actual[0])
                self.assertTrue(np.allclose(dp_expected[1], dp_actual[1]))
        except ImportError:
            print(error_msg)

    def test_lmdb(self):
        self.run_write_read_test('test.lmdb', LMDBSerializer,
                                 {}, {},
                                 {}, {'shuffle': False},
                                 'Skip test_lmdb, no lmdb available')

    def test_tfrecord(self):
        self.run_write_read_test('test.tfrecord', TFRecordSerializer,
                                 {}, {},
                                 {}, {'size': 32},
                                 'Skip test_tfrecord, no tensorflow available')

    def test_numpy(self):
        self.run_write_read_test('test.npz', NumpySerializer,
                                 {}, {},
                                 {}, {'shuffle': False},
                                 'Skip test_numpy, no numpy available')

    def test_hdf5(self):
        args = [['label', 'image']]
        self.run_write_read_test('test.h5', HDF5Serializer,
                                 args, {},
                                 args, {'shuffle': False},
                                 'Skip test_hdf5, no h5py available')


if __name__ == '__main__':
    unittest.main()
