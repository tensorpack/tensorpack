# -*- coding: utf-8 -*-
# File: config.py

import pprint

__all__ = ['config']


class AttrDict():
    def __getattr__(self, name):
        ret = AttrDict()
        setattr(self, name, ret)
        return ret

    def __str__(self):
        return pprint.pformat(self.to_dict(), indent=1)

    __repr__ = __str__

    def to_dict(self):
        """Convert to a nested dict. """
        return {k: v.to_dict() if isinstance(v, AttrDict) else v
                for k, v in self.__dict__.items()}

    def update_args(self, args):
        """Update from command line args. """
        for cfg in args:
            keys, v = cfg.split('=', maxsplit=1)
            keylist = keys.split('.')

            dic = self
            for i, k in enumerate(keylist[:-1]):
                assert k in dir(dic), "Unknown config key: {}".format(keys)
                dic = getattr(dic, k)
            key = keylist[-1]

            oldv = getattr(dic, key)
            if not isinstance(oldv, str):
                v = eval(v)
            setattr(dic, key, v)


config = AttrDict()
_C = config     # short alias to avoid coding

# mode flags ---------------------
_C.TRAINER = 'replicated'  # options: 'horovod', 'replicated'
_C.MODE_MASK = True
_C.MODE_FPN = False

# dataset -----------------------
_C.DATA.BASEDIR = '/path/to/your/COCO/DIR'
_C.DATA.TRAIN = ['train2014', 'valminusminival2014']   # i.e., trainval35k
_C.DATA.VAL = 'minival2014'   # For now, only support evaluation on single dataset
_C.DATA.NUM_CLASS = 81    # 1 background + 80 categories
_C.DATA.CLASS_NAMES = []  # NUM_CLASS strings. Needs to be populated later by data loader

# basemodel ----------------------
_C.BACKBONE.RESNET_NUM_BLOCK = [3, 4, 6, 3]     # for resnet50
# RESNET_NUM_BLOCK = [3, 4, 23, 3]    # for resnet101
_C.BACKBONE.FREEZE_AFFINE = False   # do not train affine parameters inside BN

# Use a base model with TF-preferred pad mode,
# which may pad more pixels on right/bottom than top/left.
# TF_PAD_MODE=False is better for performance but will require a different base model.
# See https://github.com/tensorflow/tensorflow/issues/18213
_C.BACKBONE.TF_PAD_MODE = True

# schedule -----------------------
# The schedule and learning rate here is defined for a total batch size of 8.
# If not running with 8 GPUs, they will be adjusted automatically in code.
_C.TRAIN.NUM_GPUS = None         # by default, will be set from code
_C.TRAIN.WEIGHT_DECAY = 1e-4
_C.TRAIN.BASE_LR = 1e-2
_C.TRAIN.WARMUP = 1000    # in steps
_C.TRAIN.STEPS_PER_EPOCH = 500
# LR_SCHEDULE = [120000, 160000, 180000]  # "1x" schedule in detectron
# LR_SCHEDULE = [150000, 230000, 280000]  # roughly a "1.5x" schedule
_C.TRAIN.LR_SCHEDULE = [240000, 320000, 360000]    # "2x" schedule in detectron

# preprocessing --------------------
_C.PREPROC.SHORT_EDGE_SIZE = 800
_C.PREPROC.MAX_SIZE = 1333
# Alternative old (worse & faster) setting: 600, 1024

# anchors -------------------------
_C.RPN.ANCHOR_STRIDE = 16
_C.RPN.ANCHOR_SIZES = (32, 64, 128, 256, 512)   # sqrtarea of the anchor box
_C.RPN.ANCHOR_RATIOS = (0.5, 1., 2.)
_C.RPN.NUM_ANCHOR = len(_C.RPN.ANCHOR_SIZES) * len(_C.RPN.ANCHOR_RATIOS)
_C.RPN.POSITIVE_ANCHOR_THRES = 0.7
_C.RPN.NEGATIVE_ANCHOR_THRES = 0.3

# rpn training -------------------------
_C.RPN.FG_RATIO = 0.5  # fg ratio among selected RPN anchors
_C.RPN.BATCH_PER_IM = 256  # total (across FPN levels) number of anchors that are marked valid
_C.RPN.MIN_SIZE = 0
_C.RPN.PROPOSAL_NMS_THRESH = 0.7
_C.RPN.TRAIN_PRE_NMS_TOPK = 12000
_C.RPN.TRAIN_POST_NMS_TOPK = 2000
_C.RPN.CROWD_OVERLAP_THRES = 0.7  # boxes overlapping crowd will be ignored.

# fastrcnn training ---------------------
_C.FRCNN.BATCH_PER_IM = 512
_C.FRCNN.BBOX_REG_WEIGHTS = [10., 10., 5., 5.]  # Better but non-standard setting: [20, 20, 10, 10]
_C.FRCNN.FG_THRESH = 0.5
_C.FRCNN.FG_RATIO = 0.25  # fg ratio in a ROI batch

# FPN -------------------------
_C.FPN.ANCHOR_STRIDES = (4, 8, 16, 32, 64)  # strides for each FPN level. Must be the same length as ANCHOR_SIZES
_C.FPN.RESOLUTION_REQUIREMENT = 32          # image size into the backbone has to be multiple of this number

_C.FPN.NUM_CHANNEL = 256
# conv head and fc head are only used in FPN.
# For C4 models, the head is C5
_C.FPN.FRCNN_HEAD_FUNC = 'fastrcnn_2fc_head'  # choices: fastrcnn_2fc_head, fastrcnn_4conv1fc_head
_C.FPN.FRCNN_CONV_HEAD_DIM = 256
_C.FPN.FRCNN_FC_HEAD_DIM = 1024

_C.RPN.TRAIN_FPN_NMS_TOPK = 2000
_C.RPN.TEST_FPN_NMS_TOPK = 1000

# Mask-RCNN
_C.MRCNN.HEAD_DIM = 256

# testing -----------------------
_C.RPN.TEST_PRE_NMS_TOPK = 6000
_C.RPN.TEST_POST_NMS_TOPK = 1000   # if you encounter OOM in inference, set this to a smaller number
_C.TEST.FRCNN_NMS_THRESH = 0.5
_C.TEST.RESULT_SCORE_THRESH = 0.05
_C.TEST.RESULT_SCORE_THRESH_VIS = 0.3   # only visualize confident results
_C.TEST.RESULTS_PER_IM = 100
