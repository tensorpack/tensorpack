# -*- coding: utf-8 -*-
# File: __init__.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import os as _os

from tensorpack.libinfo import __version__, _HAS_TF

from tensorpack.utils import *
from tensorpack.dataflow import *

# dataflow can be used alone without installing tensorflow
if _HAS_TF:
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
