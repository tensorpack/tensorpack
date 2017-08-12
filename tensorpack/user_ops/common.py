#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: common.py

from __future__ import print_function
import tensorflow as tf
import os


def compile():
    # TODO check modtime?
    include_dir = tf.sysconfig.get_include()
    file_dir = os.path.dirname(os.path.abspath(__file__))
    compile_cmd = 'INCLUDE_DIR="-isystem {}" make -C "{}"'.format(include_dir, file_dir)
    ret = os.system(compile_cmd)
    return ret


if __name__ == '__main__':
    compile()
