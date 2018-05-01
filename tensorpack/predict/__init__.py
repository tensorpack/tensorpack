#  -*- coding: utf-8 -*-
#  File: __init__.py

# https://github.com/celery/kombu/blob/7d13f9b95d0b50c94393b962e6def928511bfda6/kombu/__init__.py#L34-L36
STATICA_HACK = True
globals()['kcah_acitats'[::-1].upper()] = False
if STATICA_HACK:
    from .base import *
    from .concurrency import *
    from .config import *
    from .dataset import *
    from .multigpu import *


from pkgutil import iter_modules
import os
import os.path

__all__ = []


def global_import(name):
    p = __import__(name, globals(), locals(), level=1)
    lst = p.__all__ if '__all__' in dir(p) else dir(p)
    if lst:
        del globals()[name]
        for k in lst:
            globals()[k] = p.__dict__[k]
            __all__.append(k)


_CURR_DIR = os.path.dirname(__file__)
for _, module_name, _ in iter_modules(
        [_CURR_DIR]):
    srcpath = os.path.join(_CURR_DIR, module_name + '.py')
    if not os.path.isfile(srcpath):
        continue
    if module_name.startswith('_'):
        continue
    global_import(module_name)
