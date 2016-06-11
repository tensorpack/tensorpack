#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: common.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import os

__all__ = ['get_dataset_dir']

def get_dataset_dir(name):
    d = os.environ['TENSORPACK_DATASET']:
    if d:
        assert os.path.isdir(d)
    else:
        d = os.path.dirname(__file__)
    return os.path.join(d, name)

