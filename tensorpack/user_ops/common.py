#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: common.py

import sysconfig
import tensorflow as tf
import os

from ..utils import logger


# https://github.com/uber/horovod/blob/10835d25eccf4b198a23a0795edddf0896f6563d/horovod/tensorflow/mpi_ops.py#L30-L40
def get_ext_suffix():
    """Determine library extension for various versions of Python."""
    ext_suffix = sysconfig.get_config_var('EXT_SUFFIX')
    if ext_suffix:
        return ext_suffix

    ext_suffix = sysconfig.get_config_var('SO')
    if ext_suffix:
        return ext_suffix

    return '.so'


def compile():
    cxxflags = ' '.join(tf.sysconfig.get_compile_flags())
    ldflags = ' '.join(tf.sysconfig.get_link_flags())
    ext_suffix = get_ext_suffix()
    file_dir = os.path.dirname(os.path.abspath(__file__))
    compile_cmd = 'TF_CXXFLAGS="{}" TF_LDFLAGS="{}" EXT_SUFFIX="{}" make -C "{}"'.format(
        cxxflags, ldflags, ext_suffix, file_dir)
    logger.info("Compile user_ops by command " + compile_cmd + ' ...')
    ret = os.system(compile_cmd)
    return ret


if __name__ == '__main__':
    compile()
