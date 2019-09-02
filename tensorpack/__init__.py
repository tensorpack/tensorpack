# -*- coding: utf-8 -*-
# File: __init__.py


from tensorpack.libinfo import __version__, __git_version__, _HAS_TF

from tensorpack.utils import *
from tensorpack.dataflow import *

# dataflow can be used alone without installing tensorflow

# https://github.com/celery/kombu/blob/7d13f9b95d0b50c94393b962e6def928511bfda6/kombu/__init__.py#L34-L36
STATICA_HACK = True
globals()['kcah_acitats'[::-1].upper()] = _HAS_TF
if STATICA_HACK:
    from tensorpack.models import *

    from tensorpack.callbacks import *
    from tensorpack.tfutils import *

    from tensorpack.train import *
    from tensorpack.graph_builder import InputDesc  # kept for BC
    from tensorpack.input_source import *
    from tensorpack.predict import *
