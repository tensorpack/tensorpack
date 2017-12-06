# -*- coding: UTF-8 -*-
# File: naming.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import tensorflow as tf

GLOBAL_STEP_INCR_OP_NAME = 'global_step_incr'
GLOBAL_STEP_INCR_VAR_NAME = 'global_step_incr:0'

# extra variables to summarize during training in a moving-average way
MOVING_SUMMARY_OPS_KEY = 'MOVING_SUMMARY_OPS'

SUMMARY_BACKUP_KEYS = [tf.GraphKeys.SUMMARIES, MOVING_SUMMARY_OPS_KEY]

TRAIN_TOWER_FREEZE_KEYS = SUMMARY_BACKUP_KEYS
PREDICT_TOWER_FREEZE_KEYS = SUMMARY_BACKUP_KEYS + [tf.GraphKeys.UPDATE_OPS]
# also freeze UPDATE_OPS in inference, because they should never be used
# TODO a better way to log and warn about collection change during build_graph.
