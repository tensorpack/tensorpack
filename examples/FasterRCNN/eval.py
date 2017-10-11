#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: eval.py

import numpy as np
import tqdm
import cv2
import os
from collections import namedtuple

import tensorflow as tf
from tensorpack.dataflow import MapDataComponent, TestDataSpeed
from tensorpack.tfutils import get_default_sess_config
from tensorpack.utils.argtools import memoized
from tensorpack.utils.utils import get_tqdm_kwargs

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from coco import COCODetection, COCOMeta
from common import clip_boxes, DataFromListOfDict, CustomResize
import config

DetectionResult = namedtuple(
    'DetectionResult',
    ['class_id', 'boxes', 'scores'])


@memoized
def get_tf_nms():
    """
    Get a NMS callable.
    """
    boxes = tf.placeholder(tf.float32, shape=[None, 4])
    scores = tf.placeholder(tf.float32, shape=[None])
    indices = tf.image.non_max_suppression(
        boxes, scores,
        config.RESULTS_PER_IM, config.FASTRCNN_NMS_THRESH)
    sess = tf.Session(config=get_default_sess_config())
    return sess.make_callable(indices, [boxes, scores])


def nms_fastrcnn_results(boxes, probs):
    """
    Args:
        boxes: nx4 floatbox in float32
        probs: nxC

    Returns:
        [DetectionResult]
    """
    C = probs.shape[1]
    boxes = boxes.copy()

    boxes_per_class = {}
    nms_func = get_tf_nms()
    ret = []
    for klass in range(1, C):
        ids = np.where(probs[:, klass] > config.RESULT_SCORE_THRESH)[0]
        if ids.size == 0:
            continue
        probs_k = probs[ids, klass].flatten()
        boxes_k = boxes[ids, :]
        selected_ids = nms_func(boxes_k[:, [1, 0, 3, 2]], probs_k)
        selected_boxes = boxes_k[selected_ids, :].copy()
        ret.append(DetectionResult(klass, selected_boxes, probs_k[selected_ids]))

    if len(ret):
        newret = []
        all_scores = np.hstack([x.scores for x in ret])
        if len(all_scores) > config.RESULTS_PER_IM:
            score_thresh = np.sort(all_scores)[-config.RESULTS_PER_IM]
            for klass, boxes, scores in ret:
                keep_ids = np.where(scores >= score_thresh)[0]
                if len(keep_ids):
                    newret.append(DetectionResult(
                        klass, boxes[keep_ids, :], scores[keep_ids]))
            ret = newret
    return ret


def detect_one_image(img, model_func):
    """
    Run detection on one image, using the TF callable.
    This function should handle the preprocessing internally.

    Args:
        img: an image
        model_func: a callable from TF model, takes [image] and returns (probs, boxes)

    Returns:
        [DetectionResult]
    """
    resizer = CustomResize(config.SHORT_EDGE_SIZE, config.MAX_SIZE)
    resized_img = resizer.augment(img)
    scale = (resized_img.shape[0] * 1.0 / img.shape[0] + resized_img.shape[1] * 1.0 / img.shape[1]) / 2
    fg_probs, fg_boxes = model_func([resized_img])
    fg_boxes = fg_boxes / scale
    fg_boxes = clip_boxes(fg_boxes, img.shape[:2])
    return nms_fastrcnn_results(fg_boxes, fg_probs)


def eval_on_dataflow(df, detect_func):
    """
    Args:
        df: a DataFlow which produces (image, image_id)
        detect_func: a callable, takes [image] and returns a dict

    Returns:
        list of dict, to be dumped to COCO json format
    """
    df.reset_state()
    all_results = []
    with tqdm.tqdm(total=df.size(), **get_tqdm_kwargs()) as pbar:
        for img, img_id in df.get_data():
            results = detect_func(img)
            for classid, boxes, scores in results:
                cat_id = COCOMeta.class_id_to_category_id[classid]
                boxes[:, 2] -= boxes[:, 0]
                boxes[:, 3] -= boxes[:, 1]
                for box, score in zip(boxes, scores):
                    all_results.append({
                        'image_id': img_id,
                        'category_id': cat_id,
                        'bbox': list(map(lambda x: float(round(x, 1)), box)),
                        'score': float(round(score, 2)),
                    })
            pbar.update(1)
    return all_results


# https://github.com/pdollar/coco/blob/master/PythonAPI/pycocoEvalDemo.ipynb
def print_evaluation_scores(json_file):
    assert config.BASEDIR and os.path.isdir(config.BASEDIR)
    annofile = os.path.join(
        config.BASEDIR, 'annotations',
        'instances_{}.json'.format(config.VAL_DATASET))
    coco = COCO(annofile)
    cocoDt = coco.loadRes(json_file)
    imgIds = sorted(coco.getImgIds())
    cocoEval = COCOeval(coco, cocoDt, 'bbox')
    cocoEval.params.imgIds = imgIds
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
