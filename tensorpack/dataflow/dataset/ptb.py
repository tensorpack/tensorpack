#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: ptb.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import os
import numpy as np

from ...utils import logger, get_dataset_path
from ...utils.fs import download
from ...utils.argtools import memoized_ignoreargs
from ..base import RNGDataFlow
try:
    import tensorflow
    from tensorflow.models.rnn.ptb import reader as tfreader
except ImportError:
    logger.warn_dependency('PennTreeBank', 'tensorflow')
    __all__ = []
else:
    __all__ = ['PennTreeBank']


TRAIN_URL = 'https://raw.githubusercontent.com/tomsercu/lstm/master/data/ptb.train.txt'
VALID_URL = 'https://raw.githubusercontent.com/tomsercu/lstm/master/data/ptb.valid.txt'
TEST_URL = 'https://raw.githubusercontent.com/tomsercu/lstm/master/data/ptb.test.txt'

@memoized_ignoreargs
def get_raw_data(data_dir):
    if not os.path.isfile(os.path.join(data_dir, 'ptb.train.txt')):
        download(TRAIN_URL, data_dir)
        download(VALID_URL, data_dir)
        download(TEST_URL, data_dir)
    # TODO these functions in TF might not be available in the future
    word_to_id = tfreader._build_vocab(os.path.join(data_dir, 'ptb.train.txt'))
    data3 = [tfreader._file_to_word_ids(os.path.join(data_dir, fname), word_to_id)
            for fname in ['ptb.train.txt', 'ptb.valid.txt', 'ptb.test.txt']]
    return data3, word_to_id

class PennTreeBank(RNGDataFlow):
    def __init__(self, name, step_size, data_dir=None, shuffle=True):
        """
        Generate PTB word sequences.
        :param name: one of 'train', 'val', 'test'
        """
        super(PennTreeBank, self).__init__()
        if data_dir is None:
            data_dir = get_dataset_path('ptb_data')
        data3, word_to_id = get_raw_data(data_dir)
        self.word_to_id = word_to_id
        self.data = np.asarray(
                data3[['train', 'val', 'test'].index(name)], dtype='int32')
        self.step_size = step_size
        self.shuffle = shuffle

    def size(self):
        return (self.data.shape[0] - 1) // self.step_size

    def get_data(self):
        sz = self.size()
        if not self.shuffle:
            starts = np.arange(self.data.shape[0] - 1)[::self.step_size]
            assert starts.shape[0] >= sz
            starts = starts[:sz]
        else:
            starts = self.rng.randint(0,
                    self.data.shape[0] - 1 - self.step_size,
                    size=(sz,))
        for st in starts:
            seq = self.data[st:st+self.step_size+1]
            yield [seq[:-1],seq[1:]]

    @staticmethod
    def word_to_id():
        data3, wti = get_raw_data()
        return wti

if __name__ == '__main__':
    D = PennTreeBank('train', 50)
    D.reset_state()
    for k in D.get_data():
        import IPython as IP;
        IP.embed(config=IP.terminal.ipapp.load_default_config())

