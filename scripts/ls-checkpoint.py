#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: ls-checkpoint.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import tensorflow as tf
import numpy as np
import six
import sys
import pprint

from tensorpack.tfutils.varmanip import get_checkpoint_path

fpath = sys.argv[1]

if fpath.endswith('.npy'):
    params = np.load(fpath, encoding='latin1').item()
    dic = {k: v.shape for k, v in six.iteritems(params)}
else:
    path = get_checkpoint_path(sys.argv[1])
    reader = tf.train.NewCheckpointReader(path)
    dic = reader.get_variable_to_shape_map()
pprint.pprint(dic)
