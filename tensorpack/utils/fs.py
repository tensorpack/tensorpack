#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: fs.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import os

def mkdir_p(dirname):
    assert dirname is not None
    if dirname == '':
        return
    try:
        os.makedirs(dirname)
    except OSError as e:
        if e.errno != 17:
            raise e

