#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: naming.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

DROPOUT_PROB_OP_NAME = 'dropout_prob'
DROPOUT_PROB_VAR_NAME = 'dropout_prob:0'

GLOBAL_STEP_OP_NAME = 'global_step'
GLOBAL_STEP_VAR_NAME = 'global_step:0'

SUMMARY_WRITER_COLLECTION_KEY = 'summary_writer'

INPUT_VARS_KEY = 'INPUT_VARIABLES'
OUTPUT_VARS_KEY = 'OUTPUT_VARIABLES'
COST_VARS_KEY = 'COST_VARIABLES'        # keep track of each individual cost
SUMMARY_VARS_KEY = 'SUMMARY_VARIABLES'  # extra variables to summarize during training

# export all upper case variables
all_local_names = locals().keys()
__all__ = [x for x in all_local_names if x.upper() == x]
