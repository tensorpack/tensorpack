#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: __init__.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

from __future__ import print_function
import tensorflow as tf
import os

__all__ = ['zmq_recv']

include_dir = tf.sysconfig.get_include()
file_dir = os.path.dirname(os.path.abspath(__file__))
compile_cmd = 'INCLUDE_DIR="-isystem {}" make -C "{}"'.format(include_dir, file_dir)
print("Compiling user ops ...")
ret = os.system(compile_cmd)
if ret != 0:
    print("Failed to compile user ops!")
else:
    recv_mod = tf.load_op_library(os.path.join(file_dir, 'zmq_recv_op.so'))
    # TODO trigger recompile when load fails
    zmq_recv = recv_mod.zmq_recv
