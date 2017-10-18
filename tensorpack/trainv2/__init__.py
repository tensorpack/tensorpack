#  -*- coding: UTF-8 -*-
#  File: __init__.py
#  Author: Yuxin Wu <ppwwyyxx@gmail.com>

from pkgutil import iter_modules
import os
import os.path

__all__ = []


def global_import(name):
    p = __import__(name, globals(), locals(), level=1)
    lst = p.__all__ if '__all__' in dir(p) else []
    del globals()[name]
    for k in lst:
        globals()[k] = p.__dict__[k]
        __all__.append(k)


_CURR_DIR = os.path.dirname(__file__)
_SKIP = []
for _, module_name, _ in iter_modules(
        [_CURR_DIR]):
    srcpath = os.path.join(_CURR_DIR, module_name + '.py')
    if not os.path.isfile(srcpath):
        continue
    if module_name.startswith('_'):
        continue
    if module_name not in _SKIP:
        global_import(module_name)
