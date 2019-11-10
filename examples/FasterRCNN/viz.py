# -*- coding: utf-8 -*-
# File: viz.py

import numpy as np

from tensorpack.utils import viz
from tensorpack.utils.palette import PALETTE_RGB

from config import config as cfg
from utils.np_box_ops import area as np_area
from utils.np_box_ops import iou as np_iou
from common import polygons_to_mask


def draw_annotation(img, boxes, klass, polygons=None, is_crowd=None):
    """Will not modify img"""
    labels = []
    assert len(boxes) == len(klass)
    if is_crowd is not None:
        assert len(boxes) == len(is_crowd)
        for cls, crd in zip(klass, is_crowd):
            clsname = cfg.DATA.CLASS_NAMES[cls]
            if crd == 1:
                clsname += ';Crowd'
            labels.append(clsname)
    else:
        for cls in klass:
            labels.append(cfg.DATA.CLASS_NAMES[cls])
    img = viz.draw_boxes(img, boxes, labels)

    if polygons is not None:
        for p in polygons:
            mask = polygons_to_mask(p, img.shape[0], img.shape[1])
            img = draw_mask(img, mask)
    return img


def draw_proposal_recall(img, proposals, proposal_scores, gt_boxes):
    """
    Draw top3 proposals for each gt.
    Args:
        proposals: NPx4
        proposal_scores: NP
        gt_boxes: NG
    """
    box_ious = np_iou(gt_boxes, proposals)    # ng x np
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
    tags = ["{},{:.2f}".format(cfg.DATA.CLASS_NAMES[lb], score) for lb, score in zip(labels, scores)]
    return viz.draw_boxes(img, boxes, tags)


def draw_final_outputs(img, results):
    """
    Args:
        results: [DetectionResult]
    """
    if len(results) == 0:
        return img

    # Display in largest to smallest order to reduce occlusion
    boxes = np.asarray([r.box for r in results])
    areas = np_area(boxes)
    sorted_inds = np.argsort(-areas)

    ret = img
    tags = []

    for result_id in sorted_inds:
        r = results[result_id]
        if r.mask is not None:
            ret = draw_mask(ret, r.mask)

    for r in results:
        tags.append(
            "{},{:.2f}".format(cfg.DATA.CLASS_NAMES[r.class_id], r.score))
    ret = viz.draw_boxes(ret, boxes, tags)
    return ret


def draw_final_outputs_blackwhite(img, results):
    """
    Args:
        results: [DetectionResult]
    """
    img_bw = img.mean(axis=2)
    img_bw = np.stack([img_bw] * 3, axis=2)

    if len(results) == 0:
        return img_bw

    boxes = np.asarray([r.box for r in results])

    all_masks = [r.mask for r in results]
    if all_masks[0] is not None:
        m = all_masks[0] > 0
        for m2 in all_masks[1:]:
            m = m | (m2 > 0)
        img_bw[m] = img[m]

    tags = ["{},{:.2f}".format(cfg.DATA.CLASS_NAMES[r.class_id], r.score) for r in results]
    ret = viz.draw_boxes(img_bw, boxes, tags)
    return ret


def draw_mask(im, mask, alpha=0.5, color=None):
    """
    Overlay a mask on top of the image.

    Args:
        im: a 3-channel uint8 image in BGR
        mask: a binary 1-channel image of the same size
        color: if None, will choose automatically
    """
    if color is None:
        color = PALETTE_RGB[np.random.choice(len(PALETTE_RGB))][::-1]
    color = np.asarray(color, dtype=np.float32)
    im = np.where(np.repeat((mask > 0)[:, :, None], 3, axis=2),
                  im * (1 - alpha) + color * alpha, im)
    im = im.astype('uint8')
    return im
