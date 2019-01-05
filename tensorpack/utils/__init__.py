#  -*- coding: utf-8 -*-
#  File: __init__.py

"""
Common utils.
These utils should be irrelevant to tensorflow.
"""

# https://github.com/celery/kombu/blob/7d13f9b95d0b50c94393b962e6def928511bfda6/kombu/__init__.py#L34-L36
STATICA_HACK = True
globals()['kcah_acitats'[::-1].upper()] = False
if STATICA_HACK:
    from .utils import *


__all__ = []


def _global_import(name):
    p = __import__(name, globals(), None, level=1)
    lst = p.__all__ if '__all__' in dir(p) else dir(p)
    for k in lst:
        if not k.startswith('__'):
            globals()[k] = p.__dict__[k]
            __all__.append(k)


_global_import('utils')

# Import no other submodules. they are supposed to be explicitly imported by users.
__all__.extend(['logger'])
