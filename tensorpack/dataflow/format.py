# -*- coding: utf-8 -*-
# File: format.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import numpy as np
from six.moves import range

from ..utils import logger, get_rng, get_tqdm
from ..utils.timer import timed_operation
from ..utils.loadcaffe import get_caffe_pb
from .base import RNGDataFlow

try:
    import h5py
except ImportError:
    logger.warn("Error in 'import h5py'. HDF5Data won't be available.")
    __all__ = []
else:
    __all__ = ['HDF5Data']

try:
    import lmdb
except ImportError:
    logger.warn("Error in 'import lmdb'. LMDBData won't be available.")
else:
    __all__.extend(['LMDBData', 'CaffeLMDB', 'LMDBDataDecoder'])

try:
    import sklearn.datasets
except ImportError:
    logger.warn("Error in 'import sklearn'. SVMLightData won't be available.")
else:
    __all__.extend(['SVMLightData'])


"""
Adapters for different data format.
"""

# TODO lazy load
class HDF5Data(RNGDataFlow):
    """
    Zip data from different paths in an HDF5 file. Will load all data into memory.
    """
    def __init__(self, filename, data_paths, shuffle=True):
        """
        :param filename: h5 data file.
        :param data_paths: list of h5 paths to zipped. For example ['images', 'labels']
        :param shuffle: shuffle the order of all data.
        """
        self.f = h5py.File(filename, 'r')
        logger.info("Loading {} to memory...".format(filename))
        self.dps = [self.f[k].value for k in data_paths]
        lens = [len(k) for k in self.dps]
        assert all([k==lens[0] for k in lens])
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
    """ Read a lmdb and produce k,v pair """
    def __init__(self, lmdb_dir, shuffle=True):
        self._lmdb = lmdb.open(lmdb_dir, readonly=True, lock=False,
                map_size=1099511627776 * 2, max_readers=100)
        self._txn = self._lmdb.begin()
        self._shuffle = shuffle
        self._size = self._txn.stat()['entries']
        if shuffle:
            # get the list of keys either from __keys__ or by iterating
            self.keys = self._txn.get('__keys__')
            if not self.keys:
                self.keys = []
                with timed_operation("Loading LMDB keys ...", log_start=True), \
                        get_tqdm(total=self._size) as pbar:
                    for k in self._txn.cursor():
                        if k != '__keys__':
                            self.keys.append(k)
                            pbar.update()

    def reset_state(self):
        super(LMDBData, self).reset_state()
        self._txn = self._lmdb.begin()

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
            s = self.size()
            self.rng.shuffle(self.keys)
            for k in self.keys:
                v = self._txn.get(k)
                yield [k, v]

class LMDBDataDecoder(LMDBData):
    def __init__(self, lmdb_dir, decoder, shuffle=True):
        """
        :param decoder: a function taking k, v and return a data point,
            or return None to skip
        """
        super(LMDBDataDecoder, self).__init__(lmdb_dir, shuffle)
        self.decoder = decoder

    def get_data(self):
        for dp in super(LMDBDataDecoder, self).get_data():
            v = self.decoder(dp[0], dp[1])
            if v: yield v

class CaffeLMDB(LMDBDataDecoder):
    """ Read a Caffe LMDB file where each value contains a caffe.Datum protobuf """
    def __init__(self, lmdb_dir, shuffle=True):
        cpb = get_caffe_pb()
        def decoder(k, v):
            try:
                datum = cpb.Datum()
                datum.ParseFromString(v)
                img = np.fromstring(datum.data, dtype=np.uint8)
                img = img.reshape(datum.channels, datum.height, datum.width)
            except Exception:
                log_once("Cannot read key {}".format(k))
                return None
            return [img.transpose(1, 2, 0), datum.label]

        super(CaffeLMDB, self).__init__(
                lmdb_dir, decoder=decoder, shuffle=shuffle)

class SVMLightData(RNGDataFlow):
    """ Read X,y from a svmlight file """
    def __init__(self, filename, shuffle=True):
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
            yield [self.X[id,:], self.y[id]]
