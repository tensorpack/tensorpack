#  -*- coding: UTF-8 -*-
#  File: __init__.py
#  Author: Yuxin Wu <ppwwyyxx@gmail.com>

from pkgutil import iter_modules
import os

from .tower import get_current_tower_context, TowerContext
# don't want to include everything from .tower
__all__ = ['get_current_tower_context', 'TowerContext']


def _global_import(name):
    p = __import__(name, globals(), None, level=1)
    lst = p.__all__ if '__all__' in dir(p) else dir(p)
    for k in lst:
        if not k.startswith('__'):
            globals()[k] = p.__dict__[k]
            __all__.append(k)


_TO_IMPORT = set([
    'common',
    'sessinit',
    'argscope',
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
__all__.extend(['sessinit', 'summary', 'optimizer',
                'sesscreate', 'gradproc', 'varreplace', 'symbolic_functions',
                'distributed', 'tower'])
