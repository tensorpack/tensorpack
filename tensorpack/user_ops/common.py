#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: common.py

from __future__ import print_function
import sysconfig
import tensorflow as tf
import os


def compile():
    cxxflags = ' '.join(tf.sysconfig.get_compile_flags())
    ldflags = ' '.join(tf.sysconfig.get_link_flags())
    file_dir = os.path.dirname(os.path.abspath(__file__))
    compile_cmd = 'TF_CXXFLAGS="{}" TF_LDFLAGS="{}" make -C "{}"'.format(
        cxxflags, ldflags, file_dir)
    ret = os.system(compile_cmd)
    return ret


# https://github.com/uber/horovod/blob/10835d25eccf4b198a23a0795edddf0896f6563d/horovod/tensorflow/mpi_ops.py#L30-L40
def get_ext_suffix():
    """Determine library extension for various versions of Python."""
    return '.so'    # TODO
    ext_suffix = sysconfig.get_config_var('EXT_SUFFIX')
    if ext_suffix:
        return ext_suffix

    ext_suffix = sysconfig.get_config_var('SO')
    if ext_suffix:
        return ext_suffix

    return '.so'


if __name__ == '__main__':
    compile()
