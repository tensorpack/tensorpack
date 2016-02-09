#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: naming.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

GLOBAL_STEP_OP_NAME = 'global_step'
GLOBAL_STEP_VAR_NAME = 'global_step:0'

SUMMARY_WRITER_COLLECTION_KEY = 'summary_writer'

MOVING_SUMMARY_VARS_KEY = 'MOVING_SUMMARY_VARIABLES'  # extra variables to summarize during training
MODEL_KEY = 'MODEL'

# export all upper case variables
all_local_names = locals().keys()
__all__ = [x for x in all_local_names if x.upper() == x]
