#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# File: googlenet-iter.py
# Author: Yuxin Wu, Yuheng Zou ({wyx,zyh}@megvii.com)

import cv2
import tensorflow as tf
import argparse
import numpy as np
import multiprocessing
import msgpack
import os
import sys

from tensorpack import *
from tensorpack.tfutils.symbolic_functions import *
from tensorpack.tfutils.summary import *
from tensorpack.tfutils.varreplace import remap_variables
from quantize import get_iter_quantize


BITW = 4
BITA = 4
NUM_ITER = 4
TOTAL_BATCH_SIZE = 128
BATCH_SIZE = None


class Model(ModelDesc):
    def _get_inputs(self):
        return [InputDesc(tf.float32, [None, 224, 224, 3], 'input'),
                InputDesc(tf.int32, [None], 'label')]

    def _build_graph(self, inputs):
        image, label = inputs
        image = image / 255.0

        fw, fa = get_iter_quantize(BITW, BITA, NUM_ITER)

        old_get_variable = tf.get_variable

        # monkey-patch tf.get_variable to apply fw
        def new_get_variable(v):
            name = v.op.name
            # don't binarize first and last layer
            if not name.endswith('W') or 'conv0' in name or 'fct' in name:
                return v
            else:
                logger.info("Binarizing weight {}".format(v.op.name))
                return fw(v)

        def nonlin(x):
            if BITA == 32:
                return tf.nn.relu(x)    # still use relu for 32bit cases
            return tf.clip_by_value(x, 0.0, 1.0)

        def activate(x):
            return fa(nonlin(x * 0.1))

        def inception_bn(name, x,
                         nr_c0_conv_1x1,
                         nr_c1_conv_1x1,
                         nr_c1_conv_3x3,
                         nr_c2_conv_1x1,
                         nr_c2_conv_5x5,
                         nr_c3_conv_1x1,
                         nonlinearity=tf.nn.relu,
                         internal_nonlin=None,
                         do_proc=True):
            if internal_nonlin is None:
                internal_nonlin = nonlinearity
            outputs = []
            with tf.variable_scope(name) as scope:
                c0 = Conv2D('column_0_conv_1x1', x, nr_c0_conv_1x1, 1)
                c0 = BatchNorm('bn_0_1x1', c0)
                if do_proc:
                    c0 = activate(c0)
                outputs.append(c0)
                c1_1x1 = Conv2D('column_1_conv_1x1', x, nr_c1_conv_1x1, 1)
                c1_1x1 = BatchNorm('bn_1_1x1', c1_1x1)
                c1_1x1 = activate(c1_1x1)
                c1_3x3 = Conv2D('column_1_conv_3x3', c1_1x1, nr_c1_conv_3x3, 3)
                c1_3x3 = BatchNorm('bn_1_3x3', c1_3x3)
                if do_proc:
                    c1_3x3 = activate(c1_3x3)
                outputs.append(c1_3x3)
                c2_1x1 = Conv2D('column_2_conv_1x1', x, nr_c2_conv_1x1, 1)
                c2_1x1 = BatchNorm('bn_2_1x1', c2_1x1)
                c2_1x1 = activate(c2_1x1)
                c2_5x5 = Conv2D('column_2_conv_5x5', c2_1x1, nr_c2_conv_5x5, 5)
                c2_5x5 = BatchNorm('bn_2_5x5', c2_5x5)
                if do_proc:
                    c2_5x5 = activate(c2_5x5)
                outputs.append(c2_5x5)
                c3_maxpool = MaxPooling('column_3_maxpool', x, 3, 1, padding='SAME')
                c3_1x1 = Conv2D('column_3_conv_1x1', c3_maxpool, nr_c3_conv_1x1, 1)
                c3_1x1 = BatchNorm('bn_3_1x1', c3_1x1)
                if do_proc:
                    c3_1x1 = activate(c3_1x1)
                outputs.append(c3_1x1)
                return tf.concat(outputs, 3, name='concat')

        with remap_variables(new_get_variable), \
                argscope(BatchNorm, decay=0.9, epsilon=1e-4), \
                argscope([Conv2D, FullyConnected], use_bias=False, nl=tf.identity):
            nl = tf.identity
            l = (LinearWrap(image)
                 .Conv2D('conv1_1', 64, 7, stride=2, padding='SAME')
                 .BatchNorm('bn1_1')
                 .MaxPooling('pool1', 3, 2, padding='SAME')
                 .apply(activate)
                 .Conv2D('conv2_1', 64, 1, padding='SAME')
                 .BatchNorm('bn2_1')
                 .apply(activate)
                 .Conv2D('conv2_2', 192, 3, padding='SAME')
                 .BatchNorm('bn2_2')
                 .MaxPooling('pool2', 3, 2, padding='SAME')
                 .apply(activate)())
            l = inception_bn('inception_3_1', l, 64, 96, 128, 16, 32, 32, nl)
            l = inception_bn('inception_3_2', l, 128, 128, 192, 32, 96, 64, nl)
            l = MaxPooling('pool3', l, 3, 2, padding='SAME')
            l = inception_bn('inception_4_1', l, 192, 96, 208, 16, 48, 64, nl)
            l = inception_bn('inception_4_2', l, 160, 112, 224, 24, 64, 64, nl)
            l = inception_bn('inception_4_3', l, 128, 128, 256, 24, 64, 64, nl)
            l = inception_bn('inception_4_4', l, 112, 144, 288, 32, 64, 64, nl)
            l = inception_bn('inception_4_5', l, 256, 160, 320, 32, 128, 128, nl)
            l = MaxPooling('pool4', l, 3, 2, padding='SAME')
            l = inception_bn('inception_5_1', l, 256, 160, 320, 32, 128, 128, nl)
            l = inception_bn('inception_5_2', l, 384, 192, 384, 48, 128, 128, nl, do_proc=False)
            l = GlobalAvgPooling('gap', l)
            l = activate(l)
            l = FullyConnected('fct', l, 1000, use_bias=True)
            logits = l

        prob = tf.nn.softmax(logits, name='output')

        cost = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=label)
        cost = tf.reduce_mean(cost, name='cross_entropy_loss')

        wrong = prediction_incorrect(logits, label, 1, name='wrong-top1')
        add_moving_summary(tf.reduce_mean(wrong, name='train-error-top1'))
        wrong = prediction_incorrect(logits, label, 5, name='wrong-top5')
        add_moving_summary(tf.reduce_mean(wrong, name='train-error-top5'))

        # weight decay on all W of fc layers
        wd_cost = regularize_cost('fc.*/W', l2_regularizer(5e-6), name='regularize_cost')

        add_param_summary(('.*/W', ['histogram', 'rms']))
        self.cost = tf.add_n([cost, wd_cost], name='cost')
        add_moving_summary(cost, wd_cost, self.cost)

    def _get_optimizer(self):
        lr = get_scalar_var('learning_rate', 1e-4, summary=True)
        return tf.train.AdamOptimizer(lr, epsilon=1e-5)


def get_data(dataset_name):
    isTrain = dataset_name == 'train'
    ds = dataset.ILSVRC12(args.data, dataset_name, shuffle=isTrain)

    meta = dataset.ILSVRCMeta()
    pp_mean = meta.get_per_pixel_mean()
    pp_mean_224 = pp_mean[16:-16, 16:-16, :]

    if isTrain:
        class Resize(imgaug.ImageAugmentor):
            def __init__(self):
                self._init(locals())

            def _augment(self, img, _):
                h, w = img.shape[:2]
                size = 224
                scale = self.rng.randint(size, 308) * 1.0 / min(h, w)
                scaleX = scale * self.rng.uniform(0.85, 1.15)
                scaleY = scale * self.rng.uniform(0.85, 1.15)
                desSize = map(int, (max(size, min(w, scaleX * w)),
                                    max(size, min(h, scaleY * h))))
                dst = cv2.resize(img, tuple(desSize),
                                 interpolation=cv2.INTER_CUBIC)
                return dst

        augmentors = [
            Resize(),
            imgaug.Rotation(max_deg=10),
            imgaug.RandomApplyAug(imgaug.GaussianBlur(3), 0.5),
            imgaug.Brightness(30, True),
            imgaug.Gamma(),
            imgaug.Contrast((0.8, 1.2), True),
            imgaug.RandomCrop((224, 224)),
            imgaug.RandomApplyAug(imgaug.JpegNoise(), 0.8),
            imgaug.RandomApplyAug(imgaug.GaussianDeform(
                [(0.2, 0.2), (0.2, 0.8), (0.8, 0.8), (0.8, 0.2)],
                (224, 224), 0.2, 3), 0.1),
            imgaug.Flip(horiz=True),
            imgaug.MapImage(lambda x: x - pp_mean_224),
        ]
    else:
        def resize_func(im):
            h, w = im.shape[:2]
            scale = 256.0 / min(h, w)
            desSize = map(int, (max(224, min(w, scale * w)),
                                max(224, min(h, scale * h))))
            im = cv2.resize(im, tuple(desSize), interpolation=cv2.INTER_CUBIC)
            return im
        augmentors = [
            imgaug.MapImage(resize_func),
            imgaug.CenterCrop((224, 224)),
            imgaug.MapImage(lambda x: x - pp_mean_224),
        ]
    ds = AugmentImageComponent(ds, augmentors, copy=False)
    ds = BatchData(ds, BATCH_SIZE, remainder=not isTrain)
    if isTrain:
        ds = PrefetchDataZMQ(ds, min(12, multiprocessing.cpu_count()))
    return ds


def get_config():
    logger.auto_set_dir()
    data_train = get_data('train')
    data_test = get_data('val')

    return TrainConfig(
        dataflow=data_train,
        callbacks=[
            ModelSaver(),
            # HumanHyperParamSetter('learning_rate'),
            ScheduledHyperParamSetter(
                'learning_rate', [(0, 1e-3), (30, 1e-4), (40, 1e-5)]),
            InferenceRunner(data_test,
                            [ScalarStats('cost'),
                             ClassificationError('wrong-top1', 'val-error-top1'),
                             ClassificationError('wrong-top5', 'val-error-top5')])
        ],
        model=Model(),
        steps_per_epoch=10000,
        max_epoch=100,
    )


def run_image(model, sess_init, inputs):
    pred_config = PredictConfig(
        model=model,
        session_init=sess_init,
        input_names=['input'],
        output_names=['output']
    )
    predictor = OfflinePredictor(pred_config)
    meta = dataset.ILSVRCMeta()
    pp_mean = meta.get_per_pixel_mean()
    pp_mean_224 = pp_mean[16:-16, 16:-16, :]
    words = meta.get_synset_words_1000()

    def resize_func(im):
        h, w = im.shape[:2]
        scale = 256.0 / min(h, w)
        desSize = map(int, (max(224, min(w, scale * w)),
                            max(224, min(h, scale * h))))
        im = cv2.resize(im, tuple(desSize), interpolation=cv2.INTER_CUBIC)
        return im
    transformers = imgaug.AugmentorList([
        imgaug.MapImage(resize_func),
        imgaug.CenterCrop((224, 224)),
        imgaug.MapImage(lambda x: x - pp_mean_224),
    ])
    for f in inputs:
        assert os.path.isfile(f)
        img = cv2.imread(f).astype('float32')
        assert img is not None

        img = transformers.augment(img)[np.newaxis, :, :, :]
        outputs = predictor([img])[0]
        prob = outputs[0]
        ret = prob.argsort()[-10:][::-1]

        names = [words[i] for i in ret]
        print(f + ":")
        print(list(zip(names, prob[ret])))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='the physical ids of GPUs to use')
    parser.add_argument('--load', help='load a checkpoint, or a npy (given as the pretrained model)')
    parser.add_argument('--data', help='ILSVRC dataset dir')
    parser.add_argument('--bits',
                        help='number of bits for W,A, separated by comma', required=True)
    parser.add_argument('--iter',
                        type=int, default=4,
                        help='iters for quantize W')
    parser.add_argument('--run', help='run on a list of images with the pretrained model', nargs='*')
    args = parser.parse_args()

    BITW, BITA = map(int, args.bits.split(','))
    NUM_ITER = args.iter

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if args.run:
        assert args.load.endswith('.npy')
        run_image(Model(), DictRestore(np.load(args.load, encoding='latin1').item()), args.run)
        sys.exit()

    assert args.gpu is not None, "Need to specify a list of gpu for training!"
    NR_GPU = len(args.gpu.split(','))
    BATCH_SIZE = TOTAL_BATCH_SIZE // NR_GPU
    logger.info("Batch per tower: {}".format(BATCH_SIZE))

    config = get_config()
    if args.load:
        config.session_init = SaverRestore(args.load)
    if args.gpu:
        config.nr_tower = len(args.gpu.split(','))
    SyncMultiGPUTrainer(config).train()
