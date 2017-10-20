#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: viz.py

from six.moves import zip
import numpy as np

from tensorpack.utils import viz

from coco import COCOMeta
from utils.box_ops import get_iou_callable


def draw_annotation(img, boxes, klass, is_crowd=None):
    labels = []
    assert len(boxes) == len(klass)
    if is_crowd is not None:
        assert len(boxes) == len(is_crowd)
        for cls, crd in zip(klass, is_crowd):
            clsname = COCOMeta.class_names[cls]
            if crd == 1:
                clsname += ';Crowd'
            labels.append(clsname)
    else:
        for cls in klass:
            labels.append(COCOMeta.class_names[cls])
    img = viz.draw_boxes(img, boxes, labels)
    return img


def draw_proposal_recall(img, proposals, proposal_scores, gt_boxes):
    """
    Draw top3 proposals for each gt.
    Args:
        proposals: NPx4
        proposal_scores: NP
        gt_boxes: NG
    """
    bbox_iou_float = get_iou_callable()
    box_ious = bbox_iou_float(gt_boxes, proposals)    # ng x np
    box_ious_argsort = np.argsort(-box_ious, axis=1)
    good_proposals_ind = box_ious_argsort[:, :3]   # for each gt, find 3 best proposals
    good_proposals_ind = np.unique(good_proposals_ind.ravel())

    proposals = proposals[good_proposals_ind, :]
    tags = list(map(str, proposal_scores[good_proposals_ind]))
    img = viz.draw_boxes(img, proposals, tags)
    return img, good_proposals_ind


def draw_predictions(img, boxes, scores):
    """
    Args:
        boxes: kx4
        scores: kxC
    """
    if len(boxes) == 0:
        return img
    labels = scores.argmax(axis=1)
    scores = scores.max(axis=1)
    tags = ["{},{:.2f}".format(COCOMeta.class_names[lb], score) for lb, score in zip(labels, scores)]
    return viz.draw_boxes(img, boxes, tags)


def draw_final_outputs(img, results):
    """
    Args:
        results: [DetectionResult]
    """
    all_boxes = []
    all_tags = []
    for class_id, boxes, scores in results:
        all_boxes.extend(boxes)
        all_tags.extend(
            ["{},{:.2f}".format(COCOMeta.class_names[class_id], sc) for sc in scores])
    all_boxes = np.asarray(all_boxes)
    if all_boxes.shape[0] == 0:
        return img
    return viz.draw_boxes(img, all_boxes, all_tags)
