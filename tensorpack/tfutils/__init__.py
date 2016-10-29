#  -*- coding: UTF-8 -*-
#  File: __init__.py
#  Author: Yuxin Wu <ppwwyyxx@gmail.com>

from pkgutil import walk_packages
import os

def _global_import(name):
    p = __import__(name, globals(), None, level=1)
    lst = p.__all__ if '__all__' in dir(p) else dir(p)
    if name in ['common', 'argscope']:
        del globals()[name]
    for k in lst:
        globals()[k] = p.__dict__[k]

_global_import('sessinit')
_global_import('common')
_global_import('gradproc')
_global_import('argscope')
_global_import('tower')

