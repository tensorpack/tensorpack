# -*- coding: utf-8 -*-
# File: config.py

import numpy as np

# mode flags ---------------------
MODE_MASK = True
MODE_FPN = False

# dataset -----------------------
BASEDIR = '/path/to/your/COCO/DIR'
TRAIN_DATASET = ['train2014', 'valminusminival2014']   # i.e., trainval35k
VAL_DATASET = 'minival2014'   # For now, only support evaluation on single dataset
NUM_CLASS = 81    # 1 background + 80 categories
CLASS_NAMES = []  # NUM_CLASS strings. Needs to be populated later by data loader

# basemodel ----------------------
RESNET_NUM_BLOCK = [3, 4, 6, 3]     # for resnet50
# RESNET_NUM_BLOCK = [3, 4, 23, 3]    # for resnet101
FREEZE_AFFINE = False   # do not train affine parameters inside BN

# schedule -----------------------
BASE_LR = 1e-2
WARMUP = 1000    # in steps
STEPS_PER_EPOCH = 500
# LR_SCHEDULE = [120000, 160000, 180000]  # "1x" schedule in detectron
# LR_SCHEDULE = [150000, 230000, 280000]  # roughly a "1.5x" schedule
LR_SCHEDULE = [240000, 320000, 360000]    # "2x" schedule in detectron

# image resolution --------------------
SHORT_EDGE_SIZE = 800
MAX_SIZE = 1333
# Alternative (worse & faster) setting: 600, 1024

# anchors -------------------------
ANCHOR_STRIDE = 16
ANCHOR_STRIDES_FPN = (4, 8, 16, 32, 64)  # strides for each FPN level. Must be the same length as ANCHOR_SIZES
FPN_RESOLUTION_REQUIREMENT = 32    # image size into the backbone has to be multiple of this number
ANCHOR_SIZES = (32, 64, 128, 256, 512)   # sqrtarea of the anchor box
ANCHOR_RATIOS = (0.5, 1., 2.)
NUM_ANCHOR = len(ANCHOR_SIZES) * len(ANCHOR_RATIOS)
POSITIVE_ANCHOR_THRES = 0.7
NEGATIVE_ANCHOR_THRES = 0.3
BBOX_DECODE_CLIP = np.log(MAX_SIZE / 16.0)  # to avoid too large numbers.

# rpn training -------------------------
RPN_FG_RATIO = 0.5  # fg ratio among selected RPN anchors
RPN_BATCH_PER_IM = 256  # total (across FPN levels) number of anchors that are marked valid
RPN_MIN_SIZE = 0
RPN_PROPOSAL_NMS_THRESH = 0.7
TRAIN_PRE_NMS_TOPK = 12000
TRAIN_POST_NMS_TOPK = 2000
TRAIN_FPN_NMS_TOPK = 2000
CROWD_OVERLAP_THRES = 0.7  # boxes overlapping crowd will be ignored.

# fastrcnn training ---------------------
FASTRCNN_BATCH_PER_IM = 512
FASTRCNN_BBOX_REG_WEIGHTS = np.array([10, 10, 5, 5], dtype='float32')
FASTRCNN_FG_THRESH = 0.5
FASTRCNN_FG_RATIO = 0.25  # fg ratio in a ROI batch

# modeling -------------------------
FPN_NUM_CHANNEL = 256
FASTRCNN_FC_HEAD_DIM = 1024
MASKRCNN_HEAD_DIM = 256

# testing -----------------------
TEST_PRE_NMS_TOPK = 6000
TEST_POST_NMS_TOPK = 1000   # if you encounter OOM in inference, set this to a smaller number
TEST_FPN_NMS_TOPK = 1000
FASTRCNN_NMS_THRESH = 0.5
RESULT_SCORE_THRESH = 0.05
RESULT_SCORE_THRESH_VIS = 0.3   # only visualize confident results
RESULTS_PER_IM = 100
