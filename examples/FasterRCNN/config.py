#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: config.py

import numpy as np

# dataset -----------------------
BASEDIR = '/path/to/your/COCO/DIR'
TRAIN_DATASET = ['train2014', 'valminusminival2014']
VAL_DATASET = 'minival2014'   # only support evaluation on one dataset
NUM_CLASS = 81
CLASS_NAMES = []  # NUM_CLASS strings

# basemodel ----------------------
RESNET_NUM_BLOCK = [3, 4, 6, 3]     # resnet50

# preprocessing --------------------
SHORT_EDGE_SIZE = 600
MAX_SIZE = 1024
# alternative (better) setting: 800, 1333

# anchors -------------------------
ANCHOR_STRIDE = 16
# sqrtarea of the anchor box
ANCHOR_SIZES = (32, 64, 128, 256, 512)
ANCHOR_RATIOS = (0.5, 1., 2.)
NUM_ANCHOR = len(ANCHOR_SIZES) * len(ANCHOR_RATIOS)
POSITIVE_ANCHOR_THRES = 0.7
NEGATIVE_ANCHOR_THRES = 0.3
# just to avoid too large numbers.
BBOX_DECODE_CLIP = np.log(MAX_SIZE / 16.0)

# rpn training -------------------------
# keep fg ratio in a batch in this range
RPN_FG_RATIO = 0.5
RPN_BATCH_PER_IM = 256
RPN_MIN_SIZE = 0
RPN_PROPOSAL_NMS_THRESH = 0.7
TRAIN_PRE_NMS_TOPK = 12000
TRAIN_POST_NMS_TOPK = 2000

# boxes overlapping crowd will be ignored.
CROWD_OVERLAP_THRES = 0.7

# fastrcnn training ---------------------
FASTRCNN_BATCH_PER_IM = 256
FASTRCNN_BBOX_REG_WEIGHTS = np.array([10, 10, 5, 5], dtype='float32')
FASTRCNN_FG_THRESH = 0.5
# keep fg ratio in a batch in this range
FASTRCNN_FG_RATIO = 0.25

# testing -----------------------
TEST_PRE_NMS_TOPK = 6000
TEST_POST_NMS_TOPK = 1000
FASTRCNN_NMS_THRESH = 0.5
RESULT_SCORE_THRESH = 0.05
RESULTS_PER_IM = 100
