#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: ls-checkpoint.py

import numpy as np
import pprint
import sys
import six
import tensorflow as tf

from tensorpack.tfutils.varmanip import get_checkpoint_path

if __name__ == '__main__':
    fpath = sys.argv[1]

    if fpath.endswith('.npy'):
        params = np.load(fpath, encoding='latin1').item()
        dic = {k: v.shape for k, v in six.iteritems(params)}
    elif fpath.endswith('.npz'):
        params = dict(np.load(fpath))
        dic = {k: v.shape for k, v in six.iteritems(params)}
    else:
        path = get_checkpoint_path(fpath)
        reader = tf.train.NewCheckpointReader(path)
        dic = reader.get_variable_to_shape_map()
    pprint.pprint(dic)
