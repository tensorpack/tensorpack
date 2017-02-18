#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: ls-checkpoint.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import tensorflow as tf
import sys
import pprint

from tensorpack.tfutils.varmanip import get_checkpoint_path

path = get_checkpoint_path(sys.argv[1])
reader = tf.train.NewCheckpointReader(path)
pprint.pprint(reader.get_variable_to_shape_map())
