#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: eval.py

import numpy as np
import tqdm
import cv2
import six
import os
from collections import namedtuple, defaultdict

import tensorflow as tf
from tensorpack.dataflow import MapDataComponent, TestDataSpeed
from tensorpack.tfutils import get_default_sess_config
from tensorpack.utils.utils import get_tqdm_kwargs

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from coco import COCODetection, COCOMeta
from common import clip_boxes, DataFromListOfDict, CustomResize
import config

DetectionResult = namedtuple(
    'DetectionResult',
    ['class_id', 'boxes', 'scores'])


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

    def group_results_by_class(boxes, probs, labels):
        dic = defaultdict(list)
        for box, prob, lab in zip(boxes, probs, labels):
            dic[lab].append((box, prob))

        def mapf(lab, values):
            boxes = np.asarray([k[0] for k in values])
            probs = np.asarray([k[1] for k in values])
            return DetectionResult(lab, boxes, probs)

        return [mapf(k, v) for k, v in six.iteritems(dic)]

    resizer = CustomResize(config.SHORT_EDGE_SIZE, config.MAX_SIZE)
    resized_img = resizer.augment(img)
    scale = (resized_img.shape[0] * 1.0 / img.shape[0] + resized_img.shape[1] * 1.0 / img.shape[1]) / 2
    boxes, probs, labels = model_func(resized_img)
    boxes = boxes / scale
    boxes = clip_boxes(boxes, img.shape[:2])
    return group_results_by_class(boxes, probs, labels)


def eval_on_dataflow(df, detect_func):
    """
    Args:
        df: a DataFlow which produces (image, image_id)
        detect_func: a callable, takes [image] and returns [DetectionResult]

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
