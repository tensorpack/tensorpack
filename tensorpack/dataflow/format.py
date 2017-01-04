# -*- coding: utf-8 -*-
# File: format.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import numpy as np
from six.moves import range
import os

from ..utils import logger, get_tqdm
from ..utils.timer import timed_operation
from ..utils.loadcaffe import get_caffe_pb
from ..utils.serialize import loads
from ..utils.argtools import log_once
from .base import RNGDataFlow

try:
    import h5py
except ImportError:
    logger.warn_dependency("HDF5Data", 'h5py')
    __all__ = []
else:
    __all__ = ['HDF5Data']

try:
    import lmdb
except ImportError:
    logger.warn_dependency("LMDBData", 'lmdb')
else:
    __all__.extend(['LMDBData', 'LMDBDataDecoder', 'LMDBDataPoint', 'CaffeLMDB'])

try:
    import sklearn.datasets
except ImportError:
    logger.warn_dependency('SVMLightData', 'sklearn')
else:
    __all__.extend(['SVMLightData'])


"""
Adapters for different data format.
"""


class HDF5Data(RNGDataFlow):
    """
    Zip data from different paths in an HDF5 file.

    Warning:
        The current implementation will load all data into memory.
    """
    # TODO lazy load

    def __init__(self, filename, data_paths, shuffle=True):
        """
        Args:
            filename (str): h5 data file.
            data_paths (list): list of h5 paths to zipped.
                For example `['images', 'labels']`.
            shuffle (bool): shuffle all data.
        """
        self.f = h5py.File(filename, 'r')
        logger.info("Loading {} to memory...".format(filename))
        self.dps = [self.f[k].value for k in data_paths]
        lens = [len(k) for k in self.dps]
        assert all([k == lens[0] for k in lens])
        self._size = lens[0]
        self.shuffle = shuffle

    def size(self):
        return self._size

    def get_data(self):
        idxs = list(range(self._size))
        if self.shuffle:
            self.rng.shuffle(idxs)
        for k in idxs:
            yield [dp[k] for dp in self.dps]


class LMDBData(RNGDataFlow):
    """ Read a LMDB database and produce (k,v) pairs """
    def __init__(self, lmdb_path, shuffle=True):
        """
        Args:
            lmdb_path (str): a directory or a file.
            shuffle (bool): shuffle the keys or not.
        """
        self._lmdb_path = lmdb_path
        self._shuffle = shuffle
        self.open_lmdb()

    def open_lmdb(self):
        self._lmdb = lmdb.open(self._lmdb_path,
                               subdir=os.path.isdir(self._lmdb_path),
                               readonly=True, lock=False, readahead=False,
                               map_size=1099511627776 * 2, max_readers=100)
        self._txn = self._lmdb.begin()
        self._size = self._txn.stat()['entries']
        if self._shuffle:
            # get the list of keys either from __keys__ or by iterating
            self.keys = loads(self._txn.get('__keys__'))
            if not self.keys:
                self.keys = []
                with timed_operation("Loading LMDB keys ...", log_start=True), \
                        get_tqdm(total=self._size) as pbar:
                    for k in self._txn.cursor():
                        if k[0] != '__keys__':
                            self.keys.append(k[0])
                            pbar.update()

    def reset_state(self):
        super(LMDBData, self).reset_state()
        self.open_lmdb()

    def size(self):
        return self._size

    def get_data(self):
        if not self._shuffle:
            c = self._txn.cursor()
            while c.next():
                k, v = c.item()
                if k != '__keys__':
                    yield [k, v]
        else:
            self.rng.shuffle(self.keys)
            for k in self.keys:
                v = self._txn.get(k)
                yield [k, v]


class LMDBDataDecoder(LMDBData):
    """ Read a LMDB database and produce a decoded output."""
    def __init__(self, lmdb_path, decoder, shuffle=True):
        """
        Args:
            lmdb_path (str): a directory or a file.
            decoder (k,v -> dp | None): a function taking k, v and returning a datapoint,
                or return None to discard.
            shuffle (bool): shuffle the keys or not.
        """
        super(LMDBDataDecoder, self).__init__(lmdb_path, shuffle)
        self.decoder = decoder

    def get_data(self):
        for dp in super(LMDBDataDecoder, self).get_data():
            v = self.decoder(dp[0], dp[1])
            if v:
                yield v


class LMDBDataPoint(LMDBDataDecoder):
    """ Read a LMDB file and produce deserialized values.
        This can work with :func:`tensorpack.dataflow.dftools.dump_dataflow_to_lmdb`. """

    def __init__(self, lmdb_path, shuffle=True):
        """
        Args:
            lmdb_path (str): a directory or a file.
            shuffle (bool): shuffle the keys or not.
        """
        super(LMDBDataPoint, self).__init__(
            lmdb_path, decoder=lambda k, v: loads(v), shuffle=shuffle)


class CaffeLMDB(LMDBDataDecoder):
    """
    Read a Caffe LMDB file where each value contains a ``caffe.Datum`` protobuf.
    Produces datapoints of the format: [HWC image, label].
    """

    def __init__(self, lmdb_path, shuffle=True):
        """
        Args:
            lmdb_path (str): a directory or a file.
            shuffle (bool): shuffle the keys or not.
        """
        cpb = get_caffe_pb()

        def decoder(k, v):
            try:
                datum = cpb.Datum()
                datum.ParseFromString(v)
                img = np.fromstring(datum.data, dtype=np.uint8)
                img = img.reshape(datum.channels, datum.height, datum.width)
            except Exception:
                log_once("Cannot read key {}".format(k), 'warn')
                return None
            return [img.transpose(1, 2, 0), datum.label]

        super(CaffeLMDB, self).__init__(
            lmdb_path, decoder=decoder, shuffle=shuffle)


class SVMLightData(RNGDataFlow):
    """ Read X,y from a svmlight file, and produce [X_i, y_i] pairs. """

    def __init__(self, filename, shuffle=True):
        """
        Args:
            filename (str): input file
            shuffle (bool): shuffle the data
        """
        self.X, self.y = sklearn.datasets.load_svmlight_file(filename)
        self.X = np.asarray(self.X.todense())
        self.shuffle = shuffle

    def size(self):
        return len(self.y)

    def get_data(self):
        idxs = np.arange(self.size())
        if self.shuffle:
            self.rng.shuffle(idxs)
        for id in idxs:
            yield [self.X[id, :], self.y[id]]
