#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: _test.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import tensorflow as tf
import numpy as np
from . import *
import unittest

subs = unittest.TestCase.__subclasses__()

def run_test_case(case):
    suite = unittest.TestLoader().loadTestsFromTestCase(case)
    unittest.TextTestRunner(verbosity=2).run(suite)

for cls in subs:
    if 'tensorpack.models' in str(cls):
        run_test_case(cls)


