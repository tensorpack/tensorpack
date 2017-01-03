#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: ptb.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import os
import numpy as np

from ...utils import logger, get_dataset_path
from ...utils.fs import download
from ...utils.argtools import memoized_ignoreargs
try:
    from tensorflow.models.rnn.ptb import reader as tfreader
except ImportError:
    logger.warn_dependency('PennTreeBank', 'tensorflow')
    __all__ = []
else:
    __all__ = ['get_PennTreeBank']


TRAIN_URL = 'https://raw.githubusercontent.com/tomsercu/lstm/master/data/ptb.train.txt'
VALID_URL = 'https://raw.githubusercontent.com/tomsercu/lstm/master/data/ptb.valid.txt'
TEST_URL = 'https://raw.githubusercontent.com/tomsercu/lstm/master/data/ptb.test.txt'


@memoized_ignoreargs
def get_PennTreeBank(data_dir=None):
    if data_dir is None:
        data_dir = get_dataset_path('ptb_data')
    if not os.path.isfile(os.path.join(data_dir, 'ptb.train.txt')):
        download(TRAIN_URL, data_dir)
        download(VALID_URL, data_dir)
        download(TEST_URL, data_dir)
    # TODO these functions in TF might not be available in the future
    word_to_id = tfreader._build_vocab(os.path.join(data_dir, 'ptb.train.txt'))
    data3 = [np.asarray(tfreader._file_to_word_ids(os.path.join(data_dir, fname), word_to_id))
             for fname in ['ptb.train.txt', 'ptb.valid.txt', 'ptb.test.txt']]
    return data3, word_to_id
