# -*- coding: UTF-8 -*-
# File: __init__.py

if False:
    from .base import *
    from .convert import *
    from .crop import *
    from .deform import *
    from .geometry import *
    from .imgproc import *
    from .meta import *
    from .misc import *
    from .noise import *
    from .paste import *
    from .transform import *


import os
from pkgutil import iter_modules

__all__ = []


def global_import(name):
    p = __import__(name, globals(), locals(), level=1)
    lst = p.__all__ if '__all__' in dir(p) else dir(p)
    if lst:
        del globals()[name]
        for k in lst:
            if not k.startswith('__'):
                globals()[k] = p.__dict__[k]
                __all__.append(k)


try:
    import cv2  # noqa
except ImportError:
    from ...utils import logger
    logger.warn("Cannot import 'cv2', therefore image augmentation is not available.")
else:
    _CURR_DIR = os.path.dirname(__file__)
    for _, module_name, _ in iter_modules(
            [os.path.dirname(__file__)]):
        srcpath = os.path.join(_CURR_DIR, module_name + '.py')
        if not os.path.isfile(srcpath):
            continue
        if not module_name.startswith('_'):
            global_import(module_name)
