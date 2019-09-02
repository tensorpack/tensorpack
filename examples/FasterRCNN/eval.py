# -*- coding: utf-8 -*-
# File: eval.py

import itertools
import json
import numpy as np
import os
import sys
import tensorflow as tf
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import cv2
import pycocotools.mask as cocomask
import tqdm
from scipy import interpolate

from tensorpack.callbacks import Callback
from tensorpack.tfutils.common import get_tf_version_tuple
from tensorpack.utils import logger, get_tqdm

from common import CustomResize, clip_boxes
from config import config as cfg
from data import get_eval_dataflow
from dataset import DatasetRegistry

try:
    import horovod.tensorflow as hvd
except ImportError:
    pass


DetectionResult = namedtuple(
    'DetectionResult',
    ['box', 'score', 'class_id', 'mask'])
"""
box: 4 float
score: float
class_id: int, 1~NUM_CLASS
mask: None, or a binary image of the original image shape
"""


def _scale_box(box, scale):
    w_half = (box[2] - box[0]) * 0.5
    h_half = (box[3] - box[1]) * 0.5
    x_c = (box[2] + box[0]) * 0.5
    y_c = (box[3] + box[1]) * 0.5

    w_half *= scale
    h_half *= scale

    scaled_box = np.zeros_like(box)
    scaled_box[0] = x_c - w_half
    scaled_box[2] = x_c + w_half
    scaled_box[1] = y_c - h_half
    scaled_box[3] = y_c + h_half
    return scaled_box


def _paste_mask(box, mask, shape):
    """
    Args:
        box: 4 float
        mask: MxM floats
        shape: h,w
    Returns:
        A uint8 binary image of hxw.
    """
    assert mask.shape[0] == mask.shape[1], mask.shape

    if cfg.MRCNN.ACCURATE_PASTE:
        # This method is accurate but much slower.
        mask = np.pad(mask, [(1, 1), (1, 1)], mode='constant')
        box = _scale_box(box, float(mask.shape[0]) / (mask.shape[0] - 2))

        mask_pixels = np.arange(0.0, mask.shape[0]) + 0.5
        mask_continuous = interpolate.interp2d(mask_pixels, mask_pixels, mask, fill_value=0.0)
        h, w = shape
        ys = np.arange(0.0, h) + 0.5
        xs = np.arange(0.0, w) + 0.5
        ys = (ys - box[1]) / (box[3] - box[1]) * mask.shape[0]
        xs = (xs - box[0]) / (box[2] - box[0]) * mask.shape[1]
        # Waste a lot of compute since most indices are out-of-border
        res = mask_continuous(xs, ys)
        return (res >= 0.5).astype('uint8')
    else:
        # This method (inspired by Detectron) is less accurate but fast.

        # int() is floor
        # box fpcoor=0.0 -> intcoor=0.0
        x0, y0 = list(map(int, box[:2] + 0.5))
        # box fpcoor=h -> intcoor=h-1, inclusive
        x1, y1 = list(map(int, box[2:] - 0.5))    # inclusive
        x1 = max(x0, x1)    # require at least 1x1
        y1 = max(y0, y1)

        w = x1 + 1 - x0
        h = y1 + 1 - y0

        # rounding errors could happen here, because masks were not originally computed for this shape.
        # but it's hard to do better, because the network does not know the "original" scale
        mask = (cv2.resize(mask, (w, h)) > 0.5).astype('uint8')
        ret = np.zeros(shape, dtype='uint8')
        ret[y0:y1 + 1, x0:x1 + 1] = mask
        return ret


def predict_image(img, model_func):
    """
    Run detection on one image, using the TF callable.
    This function should handle the preprocessing internally.

    Args:
        img: an image
        model_func: a callable from the TF model.
            It takes image and returns (boxes, probs, labels, [masks])

    Returns:
        [DetectionResult]
    """
    orig_shape = img.shape[:2]
    resizer = CustomResize(cfg.PREPROC.TEST_SHORT_EDGE_SIZE, cfg.PREPROC.MAX_SIZE)
    resized_img = resizer.augment(img)
    scale = np.sqrt(resized_img.shape[0] * 1.0 / img.shape[0] * resized_img.shape[1] / img.shape[1])
    boxes, probs, labels, *masks = model_func(resized_img)

    # Some slow numpy postprocessing:
    boxes = boxes / scale
    # boxes are already clipped inside the graph, but after the floating point scaling, this may not be true any more.
    boxes = clip_boxes(boxes, orig_shape)
    if masks:
        full_masks = [_paste_mask(box, mask, orig_shape)
                      for box, mask in zip(boxes, masks[0])]
        masks = full_masks
    else:
        # fill with none
        masks = [None] * len(boxes)

    results = [DetectionResult(*args) for args in zip(boxes, probs, labels.tolist(), masks)]
    return results


def predict_dataflow(df, model_func, tqdm_bar=None):
    """
    Args:
        df: a DataFlow which produces (image, image_id)
        model_func: a callable from the TF model.
            It takes image and returns (boxes, probs, labels, [masks])
        tqdm_bar: a tqdm object to be shared among multiple evaluation instances. If None,
            will create a new one.

    Returns:
        list of dict, in the format used by
        `DatasetSplit.eval_inference_results`
    """
    df.reset_state()
    all_results = []
    with ExitStack() as stack:
        # tqdm is not quite thread-safe: https://github.com/tqdm/tqdm/issues/323
        if tqdm_bar is None:
            tqdm_bar = stack.enter_context(get_tqdm(total=df.size()))
        for img, img_id in df:
            results = predict_image(img, model_func)
            for r in results:
                # int()/float() to make it json-serializable
                res = {
                    'image_id': img_id,
                    'category_id': int(r.class_id),
                    'bbox': [round(float(x), 4) for x in r.box],
                    'score': round(float(r.score), 4),
                }

                # also append segmentation to results
                if r.mask is not None:
                    rle = cocomask.encode(
                        np.array(r.mask[:, :, None], order='F'))[0]
                    rle['counts'] = rle['counts'].decode('ascii')
                    res['segmentation'] = rle
                all_results.append(res)
            tqdm_bar.update(1)
    return all_results


def multithread_predict_dataflow(dataflows, model_funcs):
    """
    Running multiple `predict_dataflow` in multiple threads, and aggregate the results.

    Args:
        dataflows: a list of DataFlow to be used in :func:`predict_dataflow`
        model_funcs: a list of callable to be used in :func:`predict_dataflow`

    Returns:
        list of dict, in the format used by
        `DatasetSplit.eval_inference_results`
    """
    num_worker = len(model_funcs)
    assert len(dataflows) == num_worker
    if num_worker == 1:
        return predict_dataflow(dataflows[0], model_funcs[0])
    kwargs = {'thread_name_prefix': 'EvalWorker'} if sys.version_info.minor >= 6 else {}
    with ThreadPoolExecutor(max_workers=num_worker, **kwargs) as executor, \
            tqdm.tqdm(total=sum([df.size() for df in dataflows])) as pbar:
        futures = []
        for dataflow, pred in zip(dataflows, model_funcs):
            futures.append(executor.submit(predict_dataflow, dataflow, pred, pbar))
        all_results = list(itertools.chain(*[fut.result() for fut in futures]))
        return all_results


class EvalCallback(Callback):
    """
    A callback that runs evaluation once a while.
    It supports multi-gpu evaluation.
    """

    _chief_only = False

    def __init__(self, eval_dataset, in_names, out_names, output_dir):
        self._eval_dataset = eval_dataset
        self._in_names, self._out_names = in_names, out_names
        self._output_dir = output_dir

    def _setup_graph(self):
        num_gpu = cfg.TRAIN.NUM_GPUS
        if cfg.TRAINER == 'replicated':
            # TF bug in version 1.11, 1.12: https://github.com/tensorflow/tensorflow/issues/22750
            buggy_tf = get_tf_version_tuple() in [(1, 11), (1, 12)]

            # Use two predictor threads per GPU to get better throughput
            self.num_predictor = num_gpu if buggy_tf else num_gpu * 2
            self.predictors = [self._build_predictor(k % num_gpu) for k in range(self.num_predictor)]
            self.dataflows = [get_eval_dataflow(self._eval_dataset,
                                                shard=k, num_shards=self.num_predictor)
                              for k in range(self.num_predictor)]
        else:
            # Only eval on the first machine,
            # Because evaluation assumes that all horovod workers share the filesystem.
            # Alternatively, can eval on all ranks and use allgather, but allgather sometimes hangs
            self._horovod_run_eval = hvd.rank() == hvd.local_rank()
            if self._horovod_run_eval:
                self.predictor = self._build_predictor(0)
                self.dataflow = get_eval_dataflow(self._eval_dataset,
                                                  shard=hvd.local_rank(), num_shards=hvd.local_size())

            self.barrier = hvd.allreduce(tf.random_normal(shape=[1]))

    def _build_predictor(self, idx):
        return self.trainer.get_predictor(self._in_names, self._out_names, device=idx)

    def _before_train(self):
        eval_period = cfg.TRAIN.EVAL_PERIOD
        self.epochs_to_eval = set()
        for k in itertools.count(1):
            if k * eval_period > self.trainer.max_epoch:
                break
            self.epochs_to_eval.add(k * eval_period)
        self.epochs_to_eval.add(self.trainer.max_epoch)
        logger.info("[EvalCallback] Will evaluate every {} epochs".format(eval_period))

    def _eval(self):
        logdir = self._output_dir
        if cfg.TRAINER == 'replicated':
            all_results = multithread_predict_dataflow(self.dataflows, self.predictors)
        else:
            filenames = [os.path.join(
                logdir, 'outputs{}-part{}.json'.format(self.global_step, rank)
            ) for rank in range(hvd.local_size())]

            if self._horovod_run_eval:
                local_results = predict_dataflow(self.dataflow, self.predictor)
                fname = filenames[hvd.local_rank()]
                with open(fname, 'w') as f:
                    json.dump(local_results, f)
            self.barrier.eval()
            if hvd.rank() > 0:
                return
            all_results = []
            for fname in filenames:
                with open(fname, 'r') as f:
                    obj = json.load(f)
                all_results.extend(obj)
                os.unlink(fname)

        scores = DatasetRegistry.get(self._eval_dataset).eval_inference_results(all_results)
        for k, v in scores.items():
            self.trainer.monitors.put_scalar(self._eval_dataset + '-' + k, v)

    def _trigger_epoch(self):
        if self.epoch_num in self.epochs_to_eval:
            logger.info("Running evaluation ...")
            self._eval()
