#!/usr/bin/env python

import argparse
import numpy as np
import pickle
from detectron2.config import get_cfg


def convert_config(cfg):
    ret = []
    ret.append(("MODE_MASK", cfg.MODEL.MASK_ON))
    has_fpn = "fpn" in cfg.MODEL.BACKBONE.NAME
    ret.append(("MODE_FPN", has_fpn))
    if not has_fpn:
        # we only support C4 and FPN
        assert cfg.MODEL.ROI_HEADS.NAME == "Res5ROIHeads"
    else:
        ret.append(("FPN.CASCADE", cfg.MODEL.ROI_HEADS.NAME == "CascadeROIHeads"))
        assert len(cfg.MODEL.ROI_BOX_CASCADE_HEAD.IOUS) == 3
    depth = cfg.MODEL.RESNETS.DEPTH
    assert depth in [50, 101], depth
    if depth == 101:
        ret.append(("BACKBONE.RESNET_NUM_BLOCKS", [3, 4, 23, 3]))
    ret.append(("BACKBONE.STRIDE_1X1", cfg.MODEL.RESNETS.STRIDE_IN_1X1))
    ret.append(("PREPROC.PIXEL_MEAN", cfg.MODEL.PIXEL_MEAN[::-1]))
    ret.append(("PREPROC.PIXEL_STD", cfg.MODEL.PIXEL_STD[::-1]))

    assert cfg.MODEL.ROI_MASK_HEAD.POOLER_TYPE == "ROIAlignV2"
    assert cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE == "ROIAlignV2"
    return ret


def convert_weights(d, cfg):
    has_fpn = "fpn" in cfg.MODEL.BACKBONE.NAME

    ret = {}

    def _convert_conv(src, dst):
        src_w = d.pop(src + ".weight").transpose(2, 3, 1, 0)
        ret[dst + "/W"] = src_w
        if src + ".norm.weight" in d:     # has norm
            ret[dst + "/bn/gamma"] = d.pop(src + ".norm.weight")
            ret[dst + "/bn/beta"] = d.pop(src + ".norm.bias")
            ret[dst + "/bn/variance/EMA"] = d.pop(src + ".norm.running_var")
            ret[dst + "/bn/mean/EMA"] = d.pop(src + ".norm.running_mean")
        if src + ".bias" in d:
            ret[dst + "/b"] = d.pop(src + ".bias")

    def _convert_fc(src, dst):
        ret[dst + "/W"] = d.pop(src + ".weight").transpose()
        ret[dst + "/b"] = d.pop(src + ".bias")

    if has_fpn:
        backbone_prefix = "backbone.bottom_up."
    else:
        backbone_prefix = "backbone."
    _convert_conv(backbone_prefix + "stem.conv1", "conv0")
    for grpid in range(4):
        if not has_fpn and grpid == 3:
            backbone_prefix = "roi_heads."
        for blkid in range([3, 4, 6 if cfg.MODEL.RESNETS.DEPTH == 50 else 23, 3][grpid]):
            _convert_conv(backbone_prefix + f"res{grpid + 2}.{blkid}.conv1",
                          f"group{grpid}/block{blkid}/conv1")
            _convert_conv(backbone_prefix + f"res{grpid + 2}.{blkid}.conv2",
                          f"group{grpid}/block{blkid}/conv2")
            _convert_conv(backbone_prefix + f"res{grpid + 2}.{blkid}.conv3",
                          f"group{grpid}/block{blkid}/conv3")
            if blkid == 0:
                _convert_conv(backbone_prefix + f"res{grpid + 2}.{blkid}.shortcut",
                              f"group{grpid}/block{blkid}/convshortcut")

    if has_fpn:
        for lvl in range(2, 6):
            _convert_conv(f"backbone.fpn_lateral{lvl}", f"fpn/lateral_1x1_c{lvl}")
            _convert_conv(f"backbone.fpn_output{lvl}", f"fpn/posthoc_3x3_p{lvl}")

    # RPN:
    _convert_conv("proposal_generator.rpn_head.conv", "rpn/conv0")
    _convert_conv("proposal_generator.rpn_head.objectness_logits", "rpn/class")
    _convert_conv("proposal_generator.rpn_head.anchor_deltas", "rpn/box")

    def _convert_box_predictor(src, dst):
        if cfg.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG:
            _convert_fc(src + ".bbox_pred", dst + "/box")
        else:
            v = d.pop(src + ".bbox_pred.bias")
            ret[dst + "/box/b"] = np.concatenate((v[:4], v))
            v = d.pop(src + ".bbox_pred.weight")
            ret[dst + "/box/W"] = np.concatenate((v[:4, :], v), axis=0).transpose()

        _convert_fc(src + ".cls_score", dst + "/class")

        num_class = ret[dst + "/class/W"].shape[1] - 1
        idxs = np.concatenate(((num_class, ), np.arange(num_class)))
        ret[dst + "/class/W"] = ret[dst + "/class/W"][:, idxs]
        ret[dst + "/class/b"] = ret[dst + "/class/b"][idxs]

    # Fast R-CNN: box head
    has_cascade = cfg.MODEL.ROI_HEADS.NAME == "CascadeROIHeads"
    if has_cascade:
        assert cfg.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG
        for k in range(3):
            for i in range(cfg.MODEL.ROI_BOX_HEAD.NUM_FC):
                _convert_fc(f"roi_heads.box_head.{k}.fc{i+1}", f"cascade_rcnn_stage{k+1}/head/fc{i+6}")
            _convert_box_predictor(f"roi_heads.box_predictor.{k}", f"cascade_rcnn_stage{k+1}/outputs")
    else:
        for i in range(cfg.MODEL.ROI_BOX_HEAD.NUM_FC):
            _convert_fc(f"roi_heads.box_head.fc{i+1}", f"fastrcnn/fc{i+6}")
        _convert_box_predictor("roi_heads.box_predictor", "fastrcnn/outputs" if has_fpn else "fastrcnn")

    # mask head
    if cfg.MODEL.MASK_ON:
        for fcn in range(cfg.MODEL.ROI_MASK_HEAD.NUM_CONV):
            _convert_conv(f"roi_heads.mask_head.mask_fcn{fcn+1}", f"maskrcnn/fcn{fcn}")
        _convert_conv("roi_heads.mask_head.deconv", "maskrcnn/deconv")
        _convert_conv("roi_heads.mask_head.predictor", "maskrcnn/conv")

    for k in list(d.keys()):
        if "cell_anchors" in k:
            d.pop(k)
    assert len(d) == 0, d.keys()
    return ret


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--d2-config")
    parser.add_argument("--d2-pkl")
    parser.add_argument("--output")
    args = parser.parse_args()

    cfg = get_cfg()
    cfg.merge_from_file(args.d2_config)

    tp_cfg = convert_config(cfg)
    for k, v in tp_cfg:
        print("'{}={}'".format(k, v).replace(' ', ''), end=' ')

    with open(args.d2_pkl, "rb") as f:
        d2_dict = pickle.load(f)["model"]
    tp_dict = convert_weights(d2_dict, cfg)

    np.savez_compressed(args.output, **tp_dict)
