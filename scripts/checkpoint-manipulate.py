#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: checkpoint-manipulate.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>


import numpy as np
from tensorpack.tfutils.varmanip import dump_chkpt_vars
import tensorflow as tf
import sys

model_path = sys.argv[1]
reader = tf.train.NewCheckpointReader(model_path)
var_names = reader.get_variable_to_shape_map().keys()
result = {}
for n in var_names:
    result[n] = reader.get_tensor(n)
import IPython as IP; IP.embed(config=IP.terminal.ipapp.load_default_config())
