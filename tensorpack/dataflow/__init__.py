#  -*- coding: UTF-8 -*-
#  File: __init__.py
#  Author: Yuxin Wu <ppwwyyxx@gmail.com>

from pkgutil import iter_modules
import os
import os.path

__all__ = []


def _global_import(name):
    p = __import__(name, globals(), locals(), level=1)
    lst = p.__all__ if '__all__' in dir(p) else dir(p)
    del globals()[name]
    for k in lst:
        globals()[k] = p.__dict__[k]
        __all__.append(k)


__SKIP = set(['dftools', 'dataset', 'imgaug'])
for _, module_name, __ in iter_modules(
        [os.path.dirname(__file__)]):
    if not module_name.startswith('_') and \
            module_name not in __SKIP:
        _global_import(module_name)


class _LazyModule(object):
    def __init__(self, modname):
        self._modname = modname

    def __getattr__(self, name):
        dataset = __import__(self._modname, globals(), locals(), [name], 1)
        return getattr(dataset, name)


dataset = _LazyModule('dataset')
__all__.extend(['imgaug', 'dftools', 'dataset'])
