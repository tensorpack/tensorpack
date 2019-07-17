#  -*- coding: utf-8 -*-
#  File: __init__.py


from .tower import get_current_tower_context, TowerContext

# https://github.com/celery/kombu/blob/7d13f9b95d0b50c94393b962e6def928511bfda6/kombu/__init__.py#L34-L36
STATICA_HACK = True
globals()['kcah_acitats'[::-1].upper()] = False
if STATICA_HACK:
    from .common import *
    from .sessinit import *
    from .argscope import *


# don't want to include everything from .tower
__all__ = ['get_current_tower_context', 'TowerContext']


def _global_import(name):
    p = __import__(name, globals(), None, level=1)
    lst = p.__all__ if '__all__' in dir(p) else dir(p)
    for k in lst:
        if not k.startswith('__'):
            globals()[k] = p.__dict__[k]
            __all__.append(k)


_TO_IMPORT = frozenset([
    'common',
    'sessinit',
    'argscope',
])

for module_name in _TO_IMPORT:
    _global_import(module_name)

"""
TODO remove this line in the future.
Better to keep submodule names (sesscreate, varmanip, etc) out of __all__,
so that these names will be invisible under `tensorpack.` namespace.

To use these utilities, users are expected to import them explicitly, e.g.:

import tensorpack.tfutils.sessinit as sessinit
"""
__all__.extend(['sessinit', 'summary', 'optimizer',
                'sesscreate', 'gradproc', 'varreplace',
                'tower'])
