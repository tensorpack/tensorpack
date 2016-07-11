# -*- coding: utf-8 -*-
# File: format.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

from ..utils import logger, get_rng
from ..utils.timer import timed_operation
from ..utils.loadcaffe import get_caffe_pb
from .base import DataFlow

import random
from tqdm import tqdm
from six.moves import range

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
    __all__.extend(['LMDBData', 'CaffeLMDB'])


"""
Adapters for different data format.
"""

class HDF5Data(DataFlow):
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
            random.shuffle(idxs)
        for k in idxs:
            yield [dp[k] for dp in self.dps]


class LMDBData(DataFlow):
    """ Read a lmdb and produce k,v pair """
    def __init__(self, lmdb_dir, shuffle=True):
        self._lmdb = lmdb.open(lmdb_dir, readonly=True, lock=False,
                map_size=1099511627776 * 2, max_readers=100)
        self._txn = self._lmdb.begin()
        self._shuffle = shuffle
        self.rng = get_rng(self)
        self._size = self._txn.stat()['entries']
        if shuffle:
            self.keys = self._txn.get('__keys__')
            if not self.keys:
                self.keys = []
                with timed_operation("Loading LMDB keys ...", log_start=True), \
                        tqdm(total=self._size, ascii=True) as pbar:
                    for k in self._txn.cursor():
                        if k != '__keys__':
                            self.keys.append(k)
                            pbar.update()

    def reset_state(self):
        self._txn = self._lmdb.begin()
        self.rng = get_rng(self)

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
            for i in range(s):
                k = self.rng.choice(self.keys)
                v = self._txn.get(k)
                yield [k, v]

class CaffeLMDB(LMDBData):
    """ Read a Caffe LMDB file where each value contains a caffe.Datum protobuf """
    def __init__(self, lmdb_dir, shuffle=True):
        """
        :param shuffle: about 3 times slower
        """
        super(CaffeLMDB, self).__init__(lmdb_dir, shuffle)

        self.cpb = get_caffe_pb()

    def get_data(self):
        datum = self.cpb.Datum()
        def parse(k, v):
            try:
                datum.ParseFromString(v)
                img = np.fromstring(datum.data, dtype=np.uint8)
                img = img.reshape(datum.channels, datum.height, datum.width)
            except Exception:
                log_once("Cannot read key {}".format(k))
                return None
            return [img.transpose(1, 2, 0), datum.label]

        for dp in super(CaffeLMDB, self).get_data():
            v = parse(dp[0], dp[1])
            if v: yield v
