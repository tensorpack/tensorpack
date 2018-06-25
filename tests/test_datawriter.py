#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Patrick Wieschollek <mail@patwie.com>


# from tensorpack import *
from tensorpack.dataflow.base import DataFlow
from tensorpack.dataflow.serializer import LMDBDataSerializer, TFRecordDataSerializer
from tensorpack.dataflow.serializer import NumpyDataSerializer, HDF5DataSerializer
import unittest
import os
import numpy as np


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

    def size(self):
        return self._size

    def get_data(self):
        for dp in self.cache:
            yield dp


class SerializerTest(unittest.TestCase):

    def run_write_read_test(self, file, serializer, w_args, w_kwargs, r_args, r_kwargs, error_msg):
        try:
            delete_file_if_exists(file)

            ds_expected = SeededFakeDataFlow()
            serializer(file).save(ds_expected, *w_args, **w_kwargs)
            ds_actual = serializer(file).load(*r_args, **r_kwargs)

            ds_actual.reset_state()
            ds_expected.reset_state()

            for dp_expected, dp_actual in zip(ds_expected.get_data(), ds_actual.get_data()):
                self.assertEqual(dp_expected[0], dp_actual[0])
                self.assertTrue(np.allclose(dp_expected[1], dp_actual[1]))
        except ImportError:
            print(error_msg)

    def test_lmdb(self):
        self.run_write_read_test('test.lmdb', LMDBDataSerializer,
                                 {}, {'overwrite': False},
                                 {}, {'shuffle': False},
                                 'skip test_lmdb, no lmdb available')

    def test_tfrecord(self):
        self.run_write_read_test('test.tfrecord', TFRecordDataSerializer,
                                 {}, {'overwrite': False},
                                 {}, {'size': 32},
                                 'skip test_tfrecord, no tensorflow available')

    def test_numpy(self):
        self.run_write_read_test('test.npz', NumpyDataSerializer,
                                 {}, {'overwrite': False},
                                 {}, {'shuffle': False},
                                 'skip test_numpy, no numpy available')

    def test_hdf5(self):
        args = [['label', 'image']]
        self.run_write_read_test('test.npz', HDF5DataSerializer,
                                 args, {'overwrite': False},
                                 args, {'shuffle': False},
                                 'skip test_hdf5, no h5py available')


if __name__ == '__main__':
    unittest.main()
