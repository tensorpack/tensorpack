# -*- coding: utf-8 -*-
# File: dftools.py


from ..utils.develop import deprecated
from .serialize import LMDBSerializer, TFRecordSerializer

__all__ = ['dump_dataflow_to_process_queue',
           'dump_dataflow_to_lmdb', 'dump_dataflow_to_tfrecord']


from .remote import dump_dataflow_to_process_queue


@deprecated("Use LMDBSerializer.save instead!", "2019-01-31")
def dump_dataflow_to_lmdb(df, lmdb_path, write_frequency=5000):
    LMDBSerializer.save(df, lmdb_path, write_frequency)


@deprecated("Use TFRecordSerializer.save instead!", "2019-01-31")
def dump_dataflow_to_tfrecord(df, path):
    TFRecordSerializer.save(df, path)
