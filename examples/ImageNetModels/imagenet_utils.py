# -*- coding: utf-8 -*-
# File: imagenet_utils.py


import multiprocessing
import numpy as np
import os
from abc import abstractmethod
import cv2
import tensorflow as tf
import tqdm

from tensorpack import ModelDesc
from tensorpack.dataflow import AugmentImageComponent, BatchData, MultiThreadMapData, PrefetchDataZMQ, dataset, imgaug
from tensorpack.input_source import QueueInput, StagingInput
from tensorpack.models import regularize_cost
from tensorpack.predict import FeedfreePredictor, PredictConfig
from tensorpack.tfutils.summary import add_moving_summary
from tensorpack.utils import logger
from tensorpack.utils.stats import RatioCounter


"""
====== DataFlow =======
"""


def fbresnet_augmentor(isTrain):
    """
    Augmentor used in fb.resnet.torch, for BGR images in range [0,255].
    """
    interpolation = cv2.INTER_CUBIC
    # linear seems to have more stable performance.
    # but we keep cubic for compatibility with old models
    if isTrain:
        augmentors = [
            imgaug.GoogleNetRandomCropAndResize(interp=interpolation),
            # It's OK to remove the following augs if your CPU is not fast enough.
            # Removing brightness/contrast/saturation does not have a significant effect on accuracy.
            # Removing lighting leads to a tiny drop in accuracy.
            imgaug.RandomOrderAug(
                [imgaug.BrightnessScale((0.6, 1.4), clip=False),
                 imgaug.Contrast((0.6, 1.4), rgb=False, clip=False),
                 imgaug.Saturation(0.4, rgb=False),
                 # rgb-bgr conversion for the constants copied from fb.resnet.torch
                 imgaug.Lighting(0.1,
                                 eigval=np.asarray(
                                     [0.2175, 0.0188, 0.0045][::-1]) * 255.0,
                                 eigvec=np.array(
                                     [[-0.5675, 0.7192, 0.4009],
                                      [-0.5808, -0.0045, -0.8140],
                                      [-0.5836, -0.6948, 0.4203]],
                                     dtype='float32')[::-1, ::-1]
                                 )]),
            imgaug.Flip(horiz=True),
        ]
    else:
        augmentors = [
            imgaug.ResizeShortestEdge(256, interp=interpolation),
            imgaug.CenterCrop((224, 224)),
        ]
    return augmentors


def get_imagenet_dataflow(
        datadir, name, batch_size,
        augmentors=None, parallel=None):
    """
    Args:
        augmentors (list[imgaug.Augmentor]): Defaults to `fbresnet_augmentor(isTrain)`

    Returns: A DataFlow which produces BGR images and labels.

    See explanations in the tutorial:
    http://tensorpack.readthedocs.io/tutorial/efficient-dataflow.html
    """
    assert name in ['train', 'val', 'test']
    isTrain = name == 'train'
    assert datadir is not None
    if augmentors is None:
        augmentors = fbresnet_augmentor(isTrain)
    assert isinstance(augmentors, list)
    if parallel is None:
        parallel = min(40, multiprocessing.cpu_count() // 2)  # assuming hyperthreading

    if isTrain:
        ds = dataset.ILSVRC12(datadir, name, shuffle=True)
        ds = AugmentImageComponent(ds, augmentors, copy=False)
        if parallel < 16:
            logger.warn("DataFlow may become the bottleneck when too few processes are used.")
        ds = PrefetchDataZMQ(ds, parallel)
        ds = BatchData(ds, batch_size, remainder=False)
    else:
        ds = dataset.ILSVRC12Files(datadir, name, shuffle=False)
        aug = imgaug.AugmentorList(augmentors)

        def mapf(dp):
            fname, cls = dp
            im = cv2.imread(fname, cv2.IMREAD_COLOR)
            im = aug.augment(im)
            return im, cls
        ds = MultiThreadMapData(ds, parallel, mapf, buffer_size=2000, strict=True)
        ds = BatchData(ds, batch_size, remainder=True)
        ds = PrefetchDataZMQ(ds, 1)
    return ds


"""
====== tf.data =======
"""


def get_imagenet_tfdata(datadir, name, batch_size, mapper=None, parallel=None):
    """
    Args:
        mapper: a symbolic function that takes a tf.string (the raw bytes read from file) and produces a BGR image.
            Defaults to `fbresnet_mapper(isTrain)`.

    Returns:
        A `tf.data.Dataset`. If training, the dataset is infinite.
        The dataset contains BGR images and labels.
    """

    def get_imglist(dir, name):
        """
        Returns:
            [(full filename, label)]
        """
        dir = os.path.join(dir, name)
        meta = dataset.ILSVRCMeta()
        imglist = meta.get_image_list(
            name,
            dataset.ILSVRCMeta.guess_dir_structure(dir))

        def _filter(fname):
            # png
            return 'n02105855_2933.JPEG' in fname

        ret = []
        for fname, label in imglist:
            if _filter(fname):
                logger.info("Image {} was filtered out.".format(fname))
                continue
            fname = os.path.join(dir, fname)
            ret.append((fname, label))
        return ret

    assert name in ['train', 'val', 'test']
    assert datadir is not None
    isTrain = name == 'train'
    if mapper is None:
        mapper = fbresnet_mapper(isTrain)
    if parallel is None:
        parallel = min(40, multiprocessing.cpu_count() // 2)  # assuming hyperthreading
    imglist = get_imglist(datadir, name)

    N = len(imglist)
    filenames = tf.constant([k[0] for k in imglist], name='filenames')
    labels = tf.constant([k[1] for k in imglist], dtype=tf.int32, name='labels')

    ds = tf.data.Dataset.from_tensor_slices((filenames, labels))

    if isTrain:
        ds = ds.shuffle(N, reshuffle_each_iteration=True).repeat()

    ds = ds.apply(
        tf.data.experimental.map_and_batch(
            lambda fname, label: (mapper(tf.read_file(fname)), label),
            batch_size=batch_size,
            num_parallel_batches=parallel))
    ds = ds.prefetch(100)
    return ds


def fbresnet_mapper(isTrain):
    """
    Note: compared to fbresnet_augmentor, it
    lacks some photometric augmentation that may have a small effect (0.1~0.2%) on accuracy.
    """
    JPEG_OPT = {'fancy_upscaling': True, 'dct_method': 'INTEGER_ACCURATE'}

    def uint8_resize_bicubic(image, shape):
        ret = tf.image.resize_bicubic([image], shape)
        return tf.cast(tf.clip_by_value(ret, 0, 255), tf.uint8)[0]

    def resize_shortest_edge(image, image_shape, size):
        shape = tf.cast(image_shape, tf.float32)
        w_greater = tf.greater(image_shape[0], image_shape[1])
        shape = tf.cond(w_greater,
                        lambda: tf.cast([shape[0] / shape[1] * size, size], tf.int32),
                        lambda: tf.cast([size, shape[1] / shape[0] * size], tf.int32))

        return uint8_resize_bicubic(image, shape)

    def center_crop(image, size):
        image_height = tf.shape(image)[0]
        image_width = tf.shape(image)[1]

        offset_height = (image_height - size) // 2
        offset_width = (image_width - size) // 2
        image = tf.slice(image, [offset_height, offset_width, 0], [size, size, -1])
        return image

    def lighting(image, std, eigval, eigvec):
        v = tf.random_uniform(shape=[3]) * std * eigval
        inc = tf.matmul(eigvec, tf.reshape(v, [3, 1]))
        image = tf.cast(tf.cast(image, tf.float32) + tf.reshape(inc, [3]), image.dtype)
        return image

    def validation_mapper(byte):
        image = tf.image.decode_jpeg(
            tf.reshape(byte, shape=[]), 3, **JPEG_OPT)
        image = resize_shortest_edge(image, tf.shape(image), 256)
        image = center_crop(image, 224)
        image = tf.reverse(image, axis=[2])  # to BGR
        return image

    def training_mapper(byte):
        jpeg_shape = tf.image.extract_jpeg_shape(byte)  # hwc
        bbox_begin, bbox_size, distort_bbox = tf.image.sample_distorted_bounding_box(
            jpeg_shape,
            bounding_boxes=tf.zeros(shape=[0, 0, 4]),
            min_object_covered=0,
            aspect_ratio_range=[0.75, 1.33],
            area_range=[0.08, 1.0],
            max_attempts=10,
            use_image_if_no_bounding_boxes=True)

        is_bad = tf.reduce_sum(tf.cast(tf.equal(bbox_size, jpeg_shape), tf.int32)) >= 2

        def good():
            offset_y, offset_x, _ = tf.unstack(bbox_begin)
            target_height, target_width, _ = tf.unstack(bbox_size)
            crop_window = tf.stack([offset_y, offset_x, target_height, target_width])

            image = tf.image.decode_and_crop_jpeg(
                byte, crop_window, channels=3, **JPEG_OPT)
            image = uint8_resize_bicubic(image, [224, 224])
            return image

        def bad():
            image = tf.image.decode_jpeg(
                tf.reshape(byte, shape=[]), 3, **JPEG_OPT)
            image = resize_shortest_edge(image, jpeg_shape, 224)
            image = center_crop(image, 224)
            return image

        image = tf.cond(is_bad, bad, good)
        # TODO other imgproc
        image = lighting(image, 0.1,
                         eigval=np.array([0.2175, 0.0188, 0.0045], dtype='float32') * 255.0,
                         eigvec=np.array([[-0.5675, 0.7192, 0.4009],
                                          [-0.5808, -0.0045, -0.8140],
                                          [-0.5836, -0.6948, 0.4203]], dtype='float32'))
        image = tf.image.random_flip_left_right(image)
        image = tf.reverse(image, axis=[2])  # to BGR
        return image

    return training_mapper if isTrain else validation_mapper


"""
====== Model & Evaluation =======
"""


def eval_on_ILSVRC12(model, sessinit, dataflow):
    pred_config = PredictConfig(
        model=model,
        session_init=sessinit,
        input_names=['input', 'label'],
        output_names=['wrong-top1', 'wrong-top5']
    )
    acc1, acc5 = RatioCounter(), RatioCounter()

    # This does not have a visible improvement over naive predictor,
    # but will have an improvement if image_dtype is set to float32.
    pred = FeedfreePredictor(pred_config, StagingInput(QueueInput(dataflow), device='/gpu:0'))
    for _ in tqdm.trange(dataflow.size()):
        top1, top5 = pred()
        batch_size = top1.shape[0]
        acc1.feed(top1.sum(), batch_size)
        acc5.feed(top5.sum(), batch_size)

    print("Top1 Error: {}".format(acc1.ratio))
    print("Top5 Error: {}".format(acc5.ratio))


class ImageNetModel(ModelDesc):
    image_shape = 224

    """
    uint8 instead of float32 is used as input type to reduce copy overhead.
    It might hurt the performance a liiiitle bit.
    The pretrained models were trained with float32.
    """
    image_dtype = tf.uint8

    """
    Either 'NCHW' or 'NHWC'
    """
    data_format = 'NCHW'

    """
    Whether the image is BGR or RGB. If using DataFlow, then it should be BGR.
    """
    image_bgr = True

    weight_decay = 1e-4

    """
    To apply on normalization parameters, use '.*/W|.*/gamma|.*/beta'
    """
    weight_decay_pattern = '.*/W'

    """
    Scale the loss, for whatever reasons (e.g., gradient averaging, fp16 training, etc)
    """
    loss_scale = 1.

    """
    Label smoothing (See tf.losses.softmax_cross_entropy)
    """
    label_smoothing = 0.

    def inputs(self):
        return [tf.placeholder(self.image_dtype, [None, self.image_shape, self.image_shape, 3], 'input'),
                tf.placeholder(tf.int32, [None], 'label')]

    def build_graph(self, image, label):
        image = self.image_preprocess(image)
        assert self.data_format in ['NCHW', 'NHWC']
        if self.data_format == 'NCHW':
            image = tf.transpose(image, [0, 3, 1, 2])

        logits = self.get_logits(image)
        loss = ImageNetModel.compute_loss_and_error(
            logits, label, label_smoothing=self.label_smoothing)

        if self.weight_decay > 0:
            wd_loss = regularize_cost(self.weight_decay_pattern,
                                      tf.contrib.layers.l2_regularizer(self.weight_decay),
                                      name='l2_regularize_loss')
            add_moving_summary(loss, wd_loss)
            total_cost = tf.add_n([loss, wd_loss], name='cost')
        else:
            total_cost = tf.identity(loss, name='cost')
            add_moving_summary(total_cost)

        if self.loss_scale != 1.:
            logger.info("Scaling the total loss by {} ...".format(self.loss_scale))
            return total_cost * self.loss_scale
        else:
            return total_cost

    @abstractmethod
    def get_logits(self, image):
        """
        Args:
            image: 4D tensor of ``self.input_shape`` in ``self.data_format``

        Returns:
            Nx#class logits
        """

    def optimizer(self):
        lr = tf.get_variable('learning_rate', initializer=0.1, trainable=False)
        tf.summary.scalar('learning_rate-summary', lr)
        return tf.train.MomentumOptimizer(lr, 0.9, use_nesterov=True)

    def image_preprocess(self, image):
        with tf.name_scope('image_preprocess'):
            if image.dtype.base_dtype != tf.float32:
                image = tf.cast(image, tf.float32)
            mean = [0.485, 0.456, 0.406]    # rgb
            std = [0.229, 0.224, 0.225]
            if self.image_bgr:
                mean = mean[::-1]
                std = std[::-1]
            image_mean = tf.constant(mean, dtype=tf.float32) * 255.
            image_std = tf.constant(std, dtype=tf.float32) * 255.
            image = (image - image_mean) / image_std
            return image

    @staticmethod
    def compute_loss_and_error(logits, label, label_smoothing=0.):
        if label_smoothing == 0.:
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=label)
        else:
            nclass = logits.shape[-1]
            loss = tf.losses.softmax_cross_entropy(
                tf.one_hot(label, nclass),
                logits, label_smoothing=label_smoothing, reduction=tf.losses.Reduction.NONE)
        loss = tf.reduce_mean(loss, name='xentropy-loss')

        def prediction_incorrect(logits, label, topk=1, name='incorrect_vector'):
            with tf.name_scope('prediction_incorrect'):
                x = tf.logical_not(tf.nn.in_top_k(logits, label, topk))
            return tf.cast(x, tf.float32, name=name)

        wrong = prediction_incorrect(logits, label, 1, name='wrong-top1')
        add_moving_summary(tf.reduce_mean(wrong, name='train-error-top1'))

        wrong = prediction_incorrect(logits, label, 5, name='wrong-top5')
        add_moving_summary(tf.reduce_mean(wrong, name='train-error-top5'))
        return loss


if __name__ == '__main__':
    import argparse
    from tensorpack.dataflow import TestDataSpeed
    from tensorpack.tfutils import get_default_sess_config
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--aug', choices=['train', 'val'], default='val')
    parser.add_argument('--symbolic', action='store_true')
    args = parser.parse_args()

    if not args.symbolic:
        augs = fbresnet_augmentor(args.aug == 'train')
        df = get_imagenet_dataflow(
            args.data, 'train', args.batch, augs)
        # For val augmentor, Should get >100 it/s (i.e. 3k im/s) here on a decent E5 server.
        TestDataSpeed(df).start()
    else:
        assert args.aug == 'train'
        data = get_imagenet_tfdata(args.data, 'train', args.batch)

        itr = data.make_initializable_iterator()
        dp = itr.get_next()
        dpop = tf.group(*dp)
        with tf.Session(config=get_default_sess_config()) as sess:
            sess.run(itr.initializer)
            for _ in tqdm.trange(200):
                sess.run(dpop)
            for _ in tqdm.trange(5000, smoothing=0.1):
                sess.run(dpop)
