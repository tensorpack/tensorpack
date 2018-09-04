# -*- coding: utf-8 -*-

import os
import argparse
import cv2
import tensorflow as tf
from tensorpack import *
from tensorpack.tfutils.export import ServingExporter, MobileExporter

"""
This example illustrates the process of exporting a model trained in Tensorpack to:
- npz containing just the weights
- TensorFlow Serving
- TensorFlow Lite (TODO)


python3 export.py --export npz --load train_log/export/checkpoint
python3 export.py --export serving --load train_log/export/checkpoint
python3 export.py --export mobile --load train_log/export/checkpoint
"""


BATCH_SIZE = 1
SHAPE = 256
CHANNELS = 3


class Model(ModelDesc):
    """Just a simple model, which applies the Laplacian-operation to images to showcase
    the usage of variables, and alternating the inference-graph later.
    """
    def inputs(self):
        return [tf.placeholder(tf.uint8, (None, SHAPE, SHAPE, CHANNELS), 'input_img'),
                tf.placeholder(tf.uint8, (None, SHAPE, SHAPE, CHANNELS), 'target_img')]

    def build_graph(self, input_img, target_img):

        input_img = tf.cast(input_img, tf.float32)
        target_img = tf.cast(target_img, tf.float32)

        input_img = tf.image.rgb_to_grayscale(input_img)
        target_img = tf.image.rgb_to_grayscale(target_img)

        k = tf.get_variable('filter', dtype=tf.float32, initializer=[[[[0.]], [[1.]], [[0.]]], [[[1.]], [[-4.]], [[1.]]], [[[0.]], [[1.]], [[0.]]]])

        prediction_img = tf.nn.conv2d(input_img, k, strides=[1, 1, 1, 1], padding='SAME')
        self.prediction_img = tf.identity(prediction_img, name='prediction_img')

        cost = tf.losses.mean_squared_error(target_img, self.prediction_img,
                                            reduction=tf.losses.Reduction.MEAN)
        summary.add_moving_summary(cost)
        return tf.identity(cost, name='total_costs')

    def optimizer(self):
        lr = tf.get_variable('learning_rate', initializer=0.0, trainable=False)
        return tf.train.AdamOptimizer(lr)


def get_data(subset):
    """
    You might train this model using some fake data. But since the learning-rate is 0,
    it does not change the model weights.
    """
    ds = FakeData([[SHAPE, SHAPE, CHANNELS], [SHAPE, SHAPE, CHANNELS]], 1000, random=False,
                  dtype=['uint8', 'uint8'], domain=[(0, 255), (0, 10)])
    ds = BatchData(ds, BATCH_SIZE)
    return ds


def get_config():
    """
    The usual config for training.
    """
    logger.auto_set_dir()

    ds_train = get_data('train')
    ds_test = get_data('test')

    return TrainConfig(
        model=Model(),
        data=QueueInput(ds_train),
        callbacks=[
            ModelSaver(),
        ],
        steps_per_epoch=1,
        max_epoch=1,
    )


def export_npz():
    print('RUN')
    print("python tensorpack/scripts/dump-model-params.py "
          "--meta train_log/export/graph-0904-161912.meta "
          "train_log/export/checkpoint "
          "weights.npz")


class InferenceOnlyModel(Model):
    """This illustrates the way to rewrite the inference graph to accept images encoded as base64.

    Remarks:
        CloudML expects base64 encoded data to feed into placeholders with suffix "_bytes".
    """
    def inputs(self):
        # The inference graph only accepts a single image, which is different to the training model.
        return [tf.placeholder(tf.string, (None,), 'input_img_bytes')]

    def build_graph(self, input_img_bytes):
        # prepare inputs (base64 encoded strings to images)
        input_img = tf.map_fn(lambda x: tf.image.decode_png(x, channels=3), input_img_bytes, dtype=tf.uint8)
        dummy_img = tf.zeros_like(input_img)

        # It is a good idea to just copy the inference relevant parts to this graph. But to ease the
        # understanding, we use a dummy tensor for target image.
        # Note: For TensorFlow Lite we anyway remove unused graph components later
        super(InferenceOnlyModel, self).build_graph(input_img, dummy_img)

        # outputs should be base64 encoded strings agains
        prediction_img = tf.clip_by_value(self.prediction_img, 0, 255)
        prediction_img = tf.cast(prediction_img, tf.uint8)
        prediction_img_bytes = tf.map_fn(tf.image.encode_png, prediction_img, dtype=tf.string)

        # prediction_img_bytes = tf.image.encode_png(prediction_img)
        tf.identity(prediction_img_bytes, name='prediction_img_bytes')


def export_serving(model_path):
    pred_config = PredictConfig(
        session_init=get_model_loader(model_path),
        model=InferenceOnlyModel(),
        input_names=['input_img_bytes'],
        output_names=['prediction_img_bytes'])
    ServingExporter(pred_config).export('/tmp/exported')


def export_mobile(model_path):
    # pred_config = PredictConfig(
    #     session_init=get_model_loader(model_path),
    #     model=InferenceOnlyModel(),
    #     input_names=['input_img_bytes'],
    #     output_names=['prediction_img_bytes'])
    pred_config = PredictConfig(
        session_init=get_model_loader(model_path),
        model=Model(),
        input_names=['input_img'],
        output_names=['prediction_img'])
    MobileExporter(pred_config).export('/tmp/mobile_graph.pb', dtype=tf.uint8)


def apply(model_path):
    pred_config = PredictConfig(
        session_init=get_model_loader(model_path),
        model=Model(),
        input_names=['input_img'],
        output_names=['prediction_img'])

    pred = OfflinePredictor(pred_config)

    img = cv2.imread('lena.png')

    prediction = pred([img])[0]
    cv2.imwrite('applied.jpg', prediction[0])


def apply_inference_graph(model_path):
    pred_config = PredictConfig(
        session_init=get_model_loader(model_path),
        model=InferenceOnlyModel(),
        input_names=['input_img_bytes'],
        output_names=['prediction_img_bytes'])

    pred = OfflinePredictor(pred_config)

    with open('lena.png', 'rb') as f:
        buf = f.read()

    prediction = pred([buf])[0]

    with open('applied_serving.png', 'wb') as f:
        f.write(prediction[0])


def apply_mobile(graph_path):
    print("apply_mobile")
    def load_graph(fn):
        with tf.gfile.GFile(fn, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(graph_def)

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        load_graph(graph_path)

        input_img = sess.graph.get_tensor_by_name('import/input_img:0')
        prediction_img = sess.graph.get_tensor_by_name('import/prediction_img:0')

        prediction = sess.run(prediction_img, {input_img: cv2.imread('lena.png')[None, ...]})

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--load', help='load model')
    parser.add_argument('--apply', help='run sampling', default='')
    parser.add_argument('--export', help='export the model', default='')

    args = parser.parse_args()

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if args.apply != '':
        assert args.apply in ['normal', 'inference_graph', 'mobile']
        if args.apply == 'normal':
            apply(args.load)
        elif args.apply == 'serving':
            apply_inference_graph(args.load)
        else:
            apply_mobile(args.load)
    elif args.export != '':
        assert args.export in ['serving', 'mobile', 'npz']
        if args.export == 'npz':
            export_npz()
        elif args.export == 'serving':
            export_serving(args.load)
        else:
            export_mobile(args.load)
    else:
        config = get_config()

        if args.gpu:
            config.nr_tower = len(args.gpu.split(','))
        if args.load:
            config.session_init = SaverRestore(args.load)

        launch_train_with_config(config, SimpleTrainer())
