#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: eval.py

import tqdm
import os
from collections import namedtuple

from tensorpack.utils.utils import get_tqdm_kwargs

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from coco import COCOMeta
from common import clip_boxes, CustomResize
import config

DetectionResult = namedtuple(
    'DetectionResult',
    ['class_id', 'box', 'score'])
"""
class_id: int, 1~NUM_CLASS
box: 4 float
score: float
"""


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
    boxes, probs, labels = model_func(resized_img)
    boxes = boxes / scale
    boxes = clip_boxes(boxes, img.shape[:2])

    results = [DetectionResult(*args) for args in zip(labels, boxes, probs)]
    return results


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
            for classid, box, score in results:
                cat_id = COCOMeta.class_id_to_category_id[classid]
                box[2] -= box[0]
                box[3] -= box[1]
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
