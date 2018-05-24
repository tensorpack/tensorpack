# -*- coding: utf-8 -*-
# File: data.py

import cv2
import numpy as np
import copy
import itertools

from tensorpack.utils.argtools import memoized, log_once
from tensorpack.dataflow import (
    imgaug, TestDataSpeed, PrefetchDataZMQ, MultiProcessMapDataZMQ,
    MapDataComponent, DataFromList)
# import tensorpack.utils.viz as tpviz

from coco import COCODetection
from utils.generate_anchors import generate_anchors
from utils.np_box_ops import iou as np_iou
from utils.np_box_ops import area as np_area
from common import (
    DataFromListOfDict, CustomResize, filter_boxes_inside_shape,
    box_to_point8, point8_to_box, segmentation_to_mask)
import config


class MalformedData(BaseException):
    pass


@memoized
def get_all_anchors(
        stride=config.ANCHOR_STRIDE,
        sizes=config.ANCHOR_SIZES):
    """
    Get all anchors in the largest possible image, shifted, floatbox
    Args:
        stride (int): the stride of anchors.
        sizes (tuple[int]): the sizes (sqrt area) of anchors

    Returns:
        anchors: SxSxNUM_ANCHORx4, where S == ceil(MAX_SIZE/STRIDE), floatbox
        The layout in the NUM_ANCHOR dim is NUM_RATIO x NUM_SIZE.

    """
    # Generates a NAx4 matrix of anchor boxes in (x1, y1, x2, y2) format. Anchors
    # are centered on stride / 2, have (approximate) sqrt areas of the specified
    # sizes, and aspect ratios as given.
    cell_anchors = generate_anchors(
        stride,
        scales=np.array(sizes, dtype=np.float) / stride,
        ratios=np.array(config.ANCHOR_RATIOS, dtype=np.float))
    # anchors are intbox here.
    # anchors at featuremap [0,0] are centered at fpcoor (8,8) (half of stride)

    max_size = config.MAX_SIZE
    if config.MODE_FPN:
        # TODO setting this in config is perhaps better
        size_mult = config.FPN_RESOLUTION_REQUIREMENT * 1.
        max_size = np.ceil(max_size / size_mult) * size_mult
    field_size = int(np.ceil(max_size / stride))
    shifts = np.arange(0, field_size) * stride
    shift_x, shift_y = np.meshgrid(shifts, shifts)
    shift_x = shift_x.flatten()
    shift_y = shift_y.flatten()
    shifts = np.vstack((shift_x, shift_y, shift_x, shift_y)).transpose()
    # Kx4, K = field_size * field_size
    K = shifts.shape[0]

    A = cell_anchors.shape[0]
    field_of_anchors = (
        cell_anchors.reshape((1, A, 4)) +
        shifts.reshape((1, K, 4)).transpose((1, 0, 2)))
    field_of_anchors = field_of_anchors.reshape((field_size, field_size, A, 4))
    # FSxFSxAx4
    assert np.all(field_of_anchors == field_of_anchors.astype('int32'))
    field_of_anchors = field_of_anchors.astype('float32')
    field_of_anchors[:, :, :, [2, 3]] += 1
    return field_of_anchors


@memoized
def get_all_anchors_fpn(
        strides=config.ANCHOR_STRIDES_FPN,
        sizes=config.ANCHOR_SIZES):
    """
    Returns:
        [anchors]: each anchors is a SxSx NUM_ANCHOR_RATIOS x4 array.
    """
    assert len(strides) == len(sizes)
    foas = []
    for stride, size in zip(strides, sizes):
        foa = get_all_anchors(stride=stride, sizes=(size,))
        foas.append(foa)
    return foas


def get_anchor_labels(anchors, gt_boxes, crowd_boxes):
    """
    Label each anchor as fg/bg/ignore.
    Args:
        anchors: Ax4 float
        gt_boxes: Bx4 float
        crowd_boxes: Cx4 float

    Returns:
        anchor_labels: (A,) int. Each element is {-1, 0, 1}
        anchor_boxes: Ax4. Contains the target gt_box for each anchor when the anchor is fg.
    """
    # This function will modify labels and return the filtered inds
    def filter_box_label(labels, value, max_num):
        curr_inds = np.where(labels == value)[0]
        if len(curr_inds) > max_num:
            disable_inds = np.random.choice(
                curr_inds, size=(len(curr_inds) - max_num),
                replace=False)
            labels[disable_inds] = -1    # ignore them
            curr_inds = np.where(labels == value)[0]
        return curr_inds

    NA, NB = len(anchors), len(gt_boxes)
    assert NB > 0  # empty images should have been filtered already
    box_ious = np_iou(anchors, gt_boxes)  # NA x NB
    ious_argmax_per_anchor = box_ious.argmax(axis=1)  # NA,
    ious_max_per_anchor = box_ious.max(axis=1)
    ious_max_per_gt = np.amax(box_ious, axis=0, keepdims=True)  # 1xNB
    # for each gt, find all those anchors (including ties) that has the max ious with it
    anchors_with_max_iou_per_gt = np.where(box_ious == ious_max_per_gt)[0]

    # Setting NA labels: 1--fg 0--bg -1--ignore
    anchor_labels = -np.ones((NA,), dtype='int32')   # NA,

    # the order of setting neg/pos labels matter
    anchor_labels[anchors_with_max_iou_per_gt] = 1
    anchor_labels[ious_max_per_anchor >= config.POSITIVE_ANCHOR_THRES] = 1
    anchor_labels[ious_max_per_anchor < config.NEGATIVE_ANCHOR_THRES] = 0

    # First label all non-ignore candidate boxes which overlap crowd as ignore
    if crowd_boxes.size > 0:
        cand_inds = np.where(anchor_labels >= 0)[0]
        cand_anchors = anchors[cand_inds]
        ious = np_iou(cand_anchors, crowd_boxes)
        overlap_with_crowd = cand_inds[ious.max(axis=1) > config.CROWD_OVERLAP_THRES]
        anchor_labels[overlap_with_crowd] = -1

    # Subsample fg labels: ignore some fg if fg is too many
    target_num_fg = int(config.RPN_BATCH_PER_IM * config.RPN_FG_RATIO)
    fg_inds = filter_box_label(anchor_labels, 1, target_num_fg)
    if len(fg_inds) == 0:
        raise MalformedData("No valid foreground for RPN!")
    # Note that fg could be fewer than the target ratio

    # Subsample bg labels. num_bg is not allowed to be too many
    old_num_bg = np.sum(anchor_labels == 0)
    if old_num_bg == 0:
        # No valid bg/fg in this image, skip.
        # This can happen if, e.g. the image has large crowd.
        raise MalformedData("No valid background for RPN!")
    target_num_bg = config.RPN_BATCH_PER_IM - len(fg_inds)
    filter_box_label(anchor_labels, 0, target_num_bg)   # ignore return values

    # Set anchor boxes: the best gt_box for each fg anchor
    anchor_boxes = np.zeros((NA, 4), dtype='float32')
    fg_boxes = gt_boxes[ious_argmax_per_anchor[fg_inds], :]
    anchor_boxes[fg_inds, :] = fg_boxes
    return anchor_labels, anchor_boxes


def get_rpn_anchor_input(im, boxes, is_crowd):
    """
    Args:
        im: an image
        boxes: nx4, floatbox, gt. shoudn't be changed
        is_crowd: n,

    Returns:
        The anchor labels and target boxes for each pixel in the featuremap.
        fm_labels: fHxfWxNA
        fm_boxes: fHxfWxNAx4
        NA will be NUM_ANCHOR_SIZES x NUM_ANCHOR_RATIOS
    """
    boxes = boxes.copy()
    all_anchors = np.copy(get_all_anchors())
    # fHxfWxAx4 -> (-1, 4)
    featuremap_anchors_flatten = all_anchors.reshape((-1, 4))

    # only use anchors inside the image
    inside_ind, inside_anchors = filter_boxes_inside_shape(featuremap_anchors_flatten, im.shape[:2])
    # obtain anchor labels and their corresponding gt boxes
    anchor_labels, anchor_gt_boxes = get_anchor_labels(inside_anchors, boxes[is_crowd == 0], boxes[is_crowd == 1])

    # Fill them back to original size: fHxfWx1, fHxfWx4
    anchorH, anchorW = all_anchors.shape[:2]
    featuremap_labels = -np.ones((anchorH * anchorW * config.NUM_ANCHOR, ), dtype='int32')
    featuremap_labels[inside_ind] = anchor_labels
    featuremap_labels = featuremap_labels.reshape((anchorH, anchorW, config.NUM_ANCHOR))
    featuremap_boxes = np.zeros((anchorH * anchorW * config.NUM_ANCHOR, 4), dtype='float32')
    featuremap_boxes[inside_ind, :] = anchor_gt_boxes
    featuremap_boxes = featuremap_boxes.reshape((anchorH, anchorW, config.NUM_ANCHOR, 4))
    return featuremap_labels, featuremap_boxes


def get_multilevel_rpn_anchor_input(im, boxes, is_crowd):
    """
    Args:
        im: an image
        boxes: nx4, floatbox, gt. shoudn't be changed
        is_crowd: n,

    Returns:
        [(fm_labels, fm_boxes)]: Returns a tuple for each FPN level.
        Each tuple contains the anchor labels and target boxes for each pixel in the featuremap.

        fm_labels: fHxfWx NUM_ANCHOR_RATIOS
        fm_boxes: fHxfWx NUM_ANCHOR_RATIOS x4
    """
    boxes = boxes.copy()
    anchors_per_level = get_all_anchors_fpn()
    flatten_anchors_per_level = [k.reshape((-1, 4)) for k in anchors_per_level]
    all_anchors_flatten = np.concatenate(flatten_anchors_per_level, axis=0)

    inside_ind, inside_anchors = filter_boxes_inside_shape(all_anchors_flatten, im.shape[:2])
    anchor_labels, anchor_gt_boxes = get_anchor_labels(inside_anchors, boxes[is_crowd == 0], boxes[is_crowd == 1])

    # map back to all_anchors, then split to each level
    num_all_anchors = all_anchors_flatten.shape[0]
    all_labels = -np.ones((num_all_anchors, ), dtype='int32')
    all_labels[inside_ind] = anchor_labels
    all_boxes = np.zeros((num_all_anchors, 4), dtype='float32')
    all_boxes[inside_ind] = anchor_gt_boxes

    start = 0
    multilevel_inputs = []
    for level_anchor in anchors_per_level:
        assert level_anchor.shape[2] == len(config.ANCHOR_RATIOS)
        anchor_shape = level_anchor.shape[:3]   # fHxfWxNUM_ANCHOR_RATIOS
        num_anchor_this_level = np.prod(anchor_shape)
        end = start + num_anchor_this_level
        multilevel_inputs.append(
            (all_labels[start: end].reshape(anchor_shape),
             all_boxes[start: end, :].reshape(anchor_shape + (4,))
             ))
        start = end
    assert end == num_all_anchors, "{} != {}".format(end, num_all_anchors)
    return multilevel_inputs


def get_train_dataflow():
    """
    Return a training dataflow. Each datapoint consists of the following:

    An image: (h, w, 3),

    1 or more pairs of (anchor_labels, anchor_boxes):
    anchor_labels: (h', w', NA)
    anchor_boxes: (h', w', NA, 4)

    gt_boxes: (N, 4)
    gt_labels: (N,)

    If MODE_MASK, gt_masks: (N, h, w)
    """

    imgs = COCODetection.load_many(
        config.BASEDIR, config.TRAIN_DATASET, add_gt=True, add_mask=config.MODE_MASK)
    """
    To train on your own data, change this to your loader.
    Produce "imgs" as a list of dict, in the dict the following keys are needed for training:
    height, width: integer
    file_name: str
    boxes: kx4 floats
    class: k integers
    is_crowd: k booleans. Use k False if you don't know what it means.
    segmentation: k numpy arrays. Each array is a polygon of shape Nx2.
        If your segmentation annotations are masks rather than polygons,
        either convert it, or the augmentation code below will need to be
        changed or skipped accordingly.
    """

    # Valid training images should have at least one fg box.
    # But this filter shall not be applied for testing.
    imgs = list(filter(lambda img: len(img['boxes']) > 0, imgs))    # log invalid training

    ds = DataFromList(imgs, shuffle=True)

    aug = imgaug.AugmentorList(
        [CustomResize(config.SHORT_EDGE_SIZE, config.MAX_SIZE),
         imgaug.Flip(horiz=True)])

    def preprocess(img):
        fname, boxes, klass, is_crowd = img['file_name'], img['boxes'], img['class'], img['is_crowd']
        im = cv2.imread(fname, cv2.IMREAD_COLOR)
        assert im is not None, fname
        im = im.astype('float32')
        # assume floatbox as input
        assert boxes.dtype == np.float32, "Loader has to return floating point boxes!"

        # augmentation:
        im, params = aug.augment_return_params(im)
        points = box_to_point8(boxes)
        points = aug.augment_coords(points, params)
        boxes = point8_to_box(points)
        assert np.min(np_area(boxes)) > 0, "Some boxes have zero area!"

        # rpn anchor:
        try:
            if config.MODE_FPN:
                multilevel_anchor_inputs = get_multilevel_rpn_anchor_input(im, boxes, is_crowd)
                anchor_inputs = itertools.chain.from_iterable(multilevel_anchor_inputs)
            else:
                # anchor_labels, anchor_boxes
                anchor_inputs = get_rpn_anchor_input(im, boxes, is_crowd)
                assert len(anchor_inputs) == 2

            boxes = boxes[is_crowd == 0]    # skip crowd boxes in training target
            klass = klass[is_crowd == 0]
            if not len(boxes):
                raise MalformedData("No valid gt_boxes!")
        except MalformedData as e:
            log_once("Input {} is filtered for training: {}".format(fname, str(e)), 'warn')
            return None

        ret = [im] + list(anchor_inputs) + [boxes, klass]

        if config.MODE_MASK:
            # augmentation will modify the polys in-place
            segmentation = copy.deepcopy(img['segmentation'])
            segmentation = [segmentation[k] for k in range(len(segmentation)) if not is_crowd[k]]
            assert len(segmentation) == len(boxes)

            # Apply augmentation on polygon coordinates.
            # And produce one image-sized binary mask per box.
            masks = []
            for polys in segmentation:
                polys = [aug.augment_coords(p, params) for p in polys]
                masks.append(segmentation_to_mask(polys, im.shape[0], im.shape[1]))
            masks = np.asarray(masks, dtype='uint8')    # values in {0, 1}
            ret.append(masks)

            # from viz import draw_annotation, draw_mask
            # viz = draw_annotation(im, boxes, klass)
            # for mask in masks:
            #     viz = draw_mask(viz, mask)
            # tpviz.interactive_imshow(viz)
        return ret

    ds = MultiProcessMapDataZMQ(ds, 10, preprocess)
    return ds


def get_eval_dataflow():
    imgs = COCODetection.load_many(config.BASEDIR, config.VAL_DATASET, add_gt=False)
    # no filter for training
    ds = DataFromListOfDict(imgs, ['file_name', 'id'])

    def f(fname):
        im = cv2.imread(fname, cv2.IMREAD_COLOR)
        assert im is not None, fname
        return im
    ds = MapDataComponent(ds, f, 0)
    ds = PrefetchDataZMQ(ds, 1)
    return ds


if __name__ == '__main__':
    import os
    from tensorpack.dataflow import PrintData
    config.BASEDIR = os.path.expanduser('~/data/coco')
    ds = get_train_dataflow()
    ds = PrintData(ds, 100)
    TestDataSpeed(ds, 50000).start()
    ds.reset_state()
    for k in ds.get_data():
        pass
