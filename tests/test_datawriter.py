#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Patrick Wieschollek <mail@patwie.com>


# from tensorpack import *
from tensorpack.dataflow.base import DataFlow
from tensorpack.dataflow.dftools import LMDBDataWriter, TFRecordDataWriter, NumpyDataWriter, HDF5DataWriter
from tensorpack.dataflow.format import LMDBDataReader, TFRecordDataReader, NumpyDataReader, HDF5DataReader
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

    def test_lmdb(self):
        delete_file_if_exists('test.lmdb')

        ds_expected = SeededFakeDataFlow()
        serializer = LMDBDataWriter(ds_expected, 'test.lmdb')
        serializer.serialize()

        ds_actual = LMDBDataReader('test.lmdb', shuffle=False)

        ds_actual.reset_state()
        ds_expected.reset_state()

        for dp_expected, dp_actual in zip(ds_expected.get_data(), ds_actual.get_data()):
            self.assertEqual(dp_expected[0], dp_actual[0])
            self.assertTrue(np.allclose(dp_expected[1], dp_actual[1]))

    def test_tfrecord(self):
        delete_file_if_exists('test.tfrecord')

        ds_expected = SeededFakeDataFlow()
        serializer = TFRecordDataWriter(ds_expected, 'test.tfrecord')
        serializer.serialize()

        ds_actual = TFRecordDataReader('test.tfrecord', ds_expected.size())

        ds_actual.reset_state()
        ds_expected.reset_state()

        for dp_expected, dp_actual in zip(ds_expected.get_data(), ds_actual.get_data()):
            self.assertEqual(dp_expected[0], dp_actual[0])
            self.assertTrue(np.allclose(dp_expected[1], dp_actual[1]))

    def test_numpy(self):
        delete_file_if_exists('test.npz')

        ds_expected = SeededFakeDataFlow()
        serializer = NumpyDataWriter(ds_expected, 'test.npz')
        serializer.serialize()

        ds_actual = NumpyDataReader('test.npz')

        ds_actual.reset_state()
        ds_expected.reset_state()

        for dp_expected, dp_actual in zip(ds_expected.get_data(), ds_actual.get_data()):
            self.assertEqual(dp_expected[0], dp_actual[0])
            self.assertTrue(np.allclose(dp_expected[1], dp_actual[1]))

    def test_hdf5(self):
        delete_file_if_exists('test.h5')

        ds_expected = SeededFakeDataFlow()
        serializer = HDF5DataWriter(ds_expected, 'test.h5', ['label', 'image'])
        serializer.serialize()

        ds_actual = HDF5DataReader('test.h5', ['label', 'image'], shuffle=False)

        ds_actual.reset_state()
        ds_expected.reset_state()

        for dp_expected, dp_actual in zip(ds_expected.get_data(), ds_actual.get_data()):
            self.assertEqual(dp_expected[0], dp_actual[0])
            self.assertTrue(np.allclose(dp_expected[1], dp_actual[1]))


if __name__ == '__main__':
    unittest.main()
