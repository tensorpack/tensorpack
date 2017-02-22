# -*- coding: utf-8 -*-
# File: format.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import numpy as np
import six
from six.moves import range
import os

from ..utils import logger, get_tqdm
from ..utils.timer import timed_operation
from ..utils.loadcaffe import get_caffe_pb
from ..utils.serialize import loads
from ..utils.argtools import log_once
from .base import RNGDataFlow
from .common import MapData

__all__ = ['HDF5Data', 'LMDBData', 'LMDBDataDecoder', 'LMDBDataPoint',
           'CaffeLMDB', 'SVMLightData']

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
    def __init__(self, lmdb_path, shuffle=True, keys=None):
        """
        Args:
            lmdb_path (str): a directory or a file.
            shuffle (bool): shuffle the keys or not.
            keys (list[str] or str): list of str as the keys, used only when shuffle is True.
                It can also be a format string e.g. ``{:0>8d}`` which will be
                formatted with the indices from 0 to *total_size - 1*.

                If not provided, it will then look in the database for ``__keys__`` which
                :func:`dump_dataflow_to_lmdb` used to store the list of keys.
                If still not found, it will iterate over the database to find
                all the keys.
        """
        self._lmdb_path = lmdb_path
        self._shuffle = shuffle

        self.open_lmdb()
        logger.info("Found {} entries in {}".format(self._size, self._lmdb_path))
        self._set_keys(keys)

    def _set_keys(self, keys=None):
        def find_keys(txn, size):
            logger.warn("Traversing the database to find keys is slow. Your should specify the keys.")
            keys = []
            with timed_operation("Loading LMDB keys ...", log_start=True), \
                    get_tqdm(total=size) as pbar:
                for k in self._txn.cursor():
                    assert k[0] != '__keys__'
                    keys.append(k[0])
                    pbar.update()
            return keys

        if self._shuffle:
            if keys is None:
                # get the list of keys either from __keys__ or by iterating
                try:
                    self.keys = loads(self._txn.get('__keys__'))
                except Exception:
                    self.keys = find_keys(self._txn, self._size)
            else:
                # check if key-format like '{:0>8d}' was given
                if isinstance(keys, six.string_types):
                    self.keys = map(lambda x: keys.format(x), list(np.arange(self._size)))
                else:
                    self.keys = keys

    def open_lmdb(self):
        self._lmdb = lmdb.open(self._lmdb_path,
                               subdir=os.path.isdir(self._lmdb_path),
                               readonly=True, lock=False, readahead=True,
                               map_size=1099511627776 * 2, max_readers=100)
        self._txn = self._lmdb.begin()
        self._size = self._txn.stat()['entries']

    def reset_state(self):
        self._lmdb.close()
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


class LMDBDataDecoder(MapData):
    """ Read a LMDB database and produce a decoded output."""
    def __init__(self, lmdb_data, decoder):
        """
        Args:
            lmdb_data: a :class:`LMDBData` instance.
            decoder (k,v -> dp | None): a function taking k, v and returning a datapoint,
                or return None to discard.
        """
        def f(dp):
            return decoder(dp[0], dp[1])
        super(LMDBDataDecoder, self).__init__(lmdb_data, f)


class LMDBDataPoint(MapData):
    """ Read a LMDB file and produce deserialized values.
        This can work with :func:`tensorpack.dataflow.dftools.dump_dataflow_to_lmdb`. """

    def __init__(self, lmdb_data):
        """
        Args:
            lmdb_data: a :class:`LMDBData` instance.
        """
        def f(dp):
            return loads(dp[1])
        super(LMDBDataPoint, self).__init__(lmdb_data, f)


def CaffeLMDB(lmdb_path, shuffle=True, keys=None):
    """
    Read a Caffe LMDB file where each value contains a ``caffe.Datum`` protobuf.
    Produces datapoints of the format: [HWC image, label].

    Note that Caffe LMDB format is not efficient: it stores serialized raw
    arrays rather than JPEG images.

    Args:
        lmdb_path, shuffle, keys: same as :class:`LMDBData`.

    Returns:
        a :class:`LMDBDataDecoder` instance.

    Example:
        .. code-block:: python

            ds = CaffeLMDB("/tmp/validation", keys='{:0>8d}')
    """

    cpb = get_caffe_pb()
    lmdb_data = LMDBData(lmdb_path, shuffle, keys)

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
    return LMDBDataDecoder(lmdb_data, decoder)


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


from ..utils.develop import create_dummy_class   # noqa
try:
    import h5py
except ImportError:
    HDF5Data = create_dummy_class('HDF5Data', 'h5py')   # noqa

try:
    import lmdb
except ImportError:
    for klass in ['LMDBData', 'LMDBDataDecoder', 'LMDBDataPoint', 'CaffeLMDB']:
        globals()[klass] = create_dummy_class(klass, 'lmdb')

try:
    import sklearn.datasets
except ImportError:
    SVMLightData = create_dummy_class('SVMLightData', 'sklearn')   # noqa
