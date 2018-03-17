# -*- coding: utf-8 -*-
# File: __init__.py

import os as _os

from tensorpack.libinfo import __version__, _HAS_TF

from tensorpack.utils import *
from tensorpack.dataflow import *

# dataflow can be used alone without installing tensorflow
# TODO maybe separate dataflow to a new project if it's good enough

# https://github.com/celery/kombu/blob/7d13f9b95d0b50c94393b962e6def928511bfda6/kombu/__init__.py#L34-L36
STATICA_HACK = True
globals()['kcah_acitats' [::-1].upper()] = _HAS_TF
if STATICA_HACK:
    from tensorpack.models import *

    from tensorpack.callbacks import *
    from tensorpack.tfutils import *

    # Default to v2
    if _os.environ.get('TENSORPACK_TRAIN_API', 'v2') == 'v2':
        from tensorpack.train import *
    else:
        from tensorpack.trainv1 import *
    from tensorpack.graph_builder import InputDesc, ModelDesc, ModelDescBase
    from tensorpack.input_source import *
    from tensorpack.predict import *
