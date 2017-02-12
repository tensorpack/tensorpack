#  -*- coding: UTF-8 -*-
#  File: __init__.py
#  Author: Yuxin Wu <ppwwyyxx@gmail.com>

from pkgutil import iter_modules
import os

__all__ = []


def _global_import(name):
    p = __import__(name, globals(), None, level=1)
    lst = p.__all__ if '__all__' in dir(p) else dir(p)
    for k in lst:
        globals()[k] = p.__dict__[k]
        __all__.append(k)


_TO_IMPORT = set([
    'common',
    'sessinit',
    'argscope',
    'tower'
])

_CURR_DIR = os.path.dirname(__file__)
for _, module_name, _ in iter_modules(
        [_CURR_DIR]):
    srcpath = os.path.join(_CURR_DIR, module_name + '.py')
    if not os.path.isfile(srcpath):
        continue
    if module_name.startswith('_'):
        continue
    if module_name in _TO_IMPORT:
        _global_import(module_name)  # import the content to tfutils.*
    else:
        __all__.append(module_name)  # import the module separately
__all__.extend(['sessinit', 'gradproc'])
