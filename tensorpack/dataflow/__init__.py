#  -*- coding: utf-8 -*-
#  File: __init__.py

# https://github.com/celery/kombu/blob/7d13f9b95d0b50c94393b962e6def928511bfda6/kombu/__init__.py#L34-L36
STATICA_HACK = True
globals()['kcah_acitats'[::-1].upper()] = False
if STATICA_HACK:
    from .base import *
    from .common import *
    from .format import *
    from .image import *
    from .parallel_map import *
    from .parallel import *
    from .raw import *
    from .remote import *
    from .serialize import *
    from . import imgaug
    from . import dataset


from pkgutil import iter_modules
import os
import os.path
from ..utils.develop import LazyLoader

__all__ = []


def _global_import(name):
    p = __import__(name, globals(), locals(), level=1)
    lst = p.__all__ if '__all__' in dir(p) else dir(p)
    if lst:
        del globals()[name]
        for k in lst:
            if not k.startswith('__'):
                globals()[k] = p.__dict__[k]
                __all__.append(k)


__SKIP = set(['dataset', 'imgaug'])
_CURR_DIR = os.path.dirname(__file__)
for _, module_name, __ in iter_modules(
        [os.path.dirname(__file__)]):
    srcpath = os.path.join(_CURR_DIR, module_name + '.py')
    if not os.path.isfile(srcpath):
        continue
    if "_test" not in module_name and \
       not module_name.startswith('_') and \
            module_name not in __SKIP:
        _global_import(module_name)


globals()['dataset'] = LazyLoader('dataset', globals(), __name__ + '.dataset')
globals()['imgaug'] = LazyLoader('imgaug', globals(), __name__ + '.imgaug')

del LazyLoader

__all__.extend(['imgaug', 'dataset'])
