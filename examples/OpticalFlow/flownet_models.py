#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Patrick Wieschollek <mail@patwie.com>


import tensorflow as tf

from tensorpack import ModelDesc, argscope, enable_argscope_for_module

enable_argscope_for_module(tf.layers)

# FlowNet2 follows the convention of FlowNet and scales the flow prediction by
# factor 20 (note: max_displacement is 20)
DISP_SCALE = 20.


def pad(x, p=3):
    """Pad tensor in H, W

    Remarks:
        TensorFlow uses "ceil(input_spatial_shape[i] / strides[i])" rather than explicit padding
        like Caffe, pyTorch does. Hence, we need to pad here beforehand.

    Args:
        x (tf.tensor): incoming tensor
        p (int, optional): padding for H, W

    Returns:
        tf.tensor: padded tensor
    """
    return tf.pad(x, [[0, 0], [0, 0], [p, p], [p, p]])


def channel_norm(x):
    return tf.norm(x, axis=1, keep_dims=True)


def correlation(ina, inb,
                kernel_size, max_displacement,
                stride_1, stride_2,
                pad, data_format):
    """
    Correlation Cost Volume computation.

    This is a fallback Python-only implementation, specialized just for FlowNet2.
    It takes a lot of memory and is slow.

    If you know to compile a custom op yourself, it's better to use the cuda implementation here:
    https://github.com/PatWie/tensorflow-recipes/tree/master/OpticalFlow/user_ops
    """
    assert pad == max_displacement
    assert kernel_size == 1
    assert data_format == 'NCHW'
    assert max_displacement % stride_2 == 0
    assert stride_1 == 1

    D = int(max_displacement / stride_2 * 2) + 1  # D^2 == number of correlations per spatial location

    b, c, h, w = ina.shape.as_list()

    inb = tf.pad(inb, [[0, 0], [0, 0], [pad, pad], [pad, pad]])

    res = []
    for k1 in range(0, D):
        start_h = k1 * stride_2
        for k2 in range(0, D):
            start_w = k2 * stride_2
            s = tf.slice(inb, [0, 0, start_h, start_w], [-1, -1, h, w])
            ans = tf.reduce_mean(ina * s, axis=1, keepdims=True)
            res.append(ans)
    res = tf.concat(res, axis=1)   # ND^2HW
    return res


def resample(img, flow):
    # img, NCHW
    # flow, N2HW
    B = tf.shape(img)[0]
    c = tf.shape(img)[1]
    h = tf.shape(img)[2]
    w = tf.shape(img)[3]
    img_flat = tf.reshape(tf.transpose(img, [0, 2, 3, 1]), [-1, c])

    dx, dy = tf.unstack(flow, axis=1)
    xf, yf = tf.meshgrid(tf.cast(tf.range(w), tf.float32), tf.cast(tf.range(h), tf.float32))
    xf = xf + dx
    yf = yf + dy

    alpha = tf.expand_dims(xf - tf.floor(xf), axis=-1)
    beta = tf.expand_dims(yf - tf.floor(yf), axis=-1)

    xL = tf.clip_by_value(tf.cast(tf.floor(xf), dtype=tf.int32), 0, w - 1)
    xR = tf.clip_by_value(tf.cast(tf.floor(xf) + 1, dtype=tf.int32), 0, w - 1)
    yT = tf.clip_by_value(tf.cast(tf.floor(yf), dtype=tf.int32), 0, h - 1)
    yB = tf.clip_by_value(tf.cast(tf.floor(yf) + 1, dtype=tf.int32), 0, h - 1)

    batch_ids = tf.tile(tf.expand_dims(tf.expand_dims(tf.range(B), axis=-1), axis=-1), [1, h, w])

    def get(y, x):
        idx = tf.reshape(batch_ids * h * w + y * w + x, [-1])
        idx = tf.cast(idx, tf.int32)
        return tf.gather(img_flat, idx)

    val = tf.zeros_like(alpha)
    val += (1 - alpha) * (1 - beta) * tf.reshape(get(yT, xL), [-1, h, w, c])
    val += (0 + alpha) * (1 - beta) * tf.reshape(get(yT, xR), [-1, h, w, c])
    val += (1 - alpha) * (0 + beta) * tf.reshape(get(yB, xL), [-1, h, w, c])
    val += (0 + alpha) * (0 + beta) * tf.reshape(get(yB, xR), [-1, h, w, c])

    # we need to enforce the channel_dim known during compile-time here
    shp = img.shape.as_list()
    return tf.reshape(tf.transpose(val, [0, 3, 1, 2]), [-1, shp[1], h, w])


def resize(x, mode, factor=4):
    """Resize input tensor with unkown input-shape by a factor

    Args:
        x (tf.Tensor): tensor NCHW
        factor (int, optional): resize factor for H, W

    Note:
        Differences here against Caffe have huge impacts on the
        quality of the predictions.

    Returns:
        tf.Tensor: resized tensor NCHW
    """
    assert mode in ['bilinear', 'nearest'], mode
    shp = tf.shape(x)[2:] * factor
    # NCHW -> NHWC
    x = tf.transpose(x, [0, 2, 3, 1])
    if mode == 'bilinear':
        x = tf.image.resize_bilinear(x, shp, align_corners=True)
    else:
        # better approximation of what Caffe is doing
        x = tf.image.resize_nearest_neighbor(x, shp, align_corners=False)
    # NHWC -> NCHW
    return tf.transpose(x, [0, 3, 1, 2])


class FlowNetBase(ModelDesc):
    def __init__(self, height=None, width=None):
        self.height = height
        self.width = width

    def inputs(self):
        return [tf.placeholder(tf.float32, (1, 3, self.height, self.width), 'left'),
                tf.placeholder(tf.float32, (1, 3, self.height, self.width), 'right'),
                tf.placeholder(tf.float32, (1, 2, self.height, self.width), 'gt_flow')]

    def graph_structure(self, inputs):
        """
        Args:
            inputs: [2, C, H, W]
        """
        raise NotImplementedError()

    def preprocess(self, left, right):
        x = tf.concat([left, right], axis=0)  # 2CHW
        rgb_mean = tf.reduce_mean(x, axis=[0, 2, 3], keep_dims=True)
        return (x - rgb_mean) / 255.

    def postprocess(self, prediction):
        return resize(prediction * DISP_SCALE, mode='bilinear')

    def build_graph(self, left, right, gt_flow):
        x = self.preprocess(left, right)
        prediction = self.graph_structure(x)
        prediction = self.postprocess(prediction)
        tf.identity(prediction, name="prediction")
        # endpoint error
        tf.reduce_mean(tf.norm(prediction - gt_flow, axis=1), name='epe')


class FlowNet2(FlowNetBase):

    def postprocess(self, prediction):
        return prediction

    def graph_structure(self, x):
        x1, x2 = tf.split(x, 2, axis=0)
        x1x2 = tf.reshape(x, [1, -1, self.height, self.width])  # 1(2C)HW

        # FlowNet-C
        flownetc_flow2 = FlowNet2C().graph_structure(x)
        flownetc_flow = resize(flownetc_flow2 * DISP_SCALE, mode='bilinear')

        resampled_img1 = resample(x2, flownetc_flow)
        norm_diff_img0 = channel_norm(x1 - resampled_img1)

        # FlowNet-S
        concat1 = tf.concat([x1x2, resampled_img1, flownetc_flow / DISP_SCALE, norm_diff_img0], axis=1)
        with tf.variable_scope('flownet_s1'):
            flownets1_flow2 = FlowNet2S().graph_structure(concat1, standalone=False)
        flownets1_flow = resize(flownets1_flow2 * DISP_SCALE, mode='bilinear')

        resampled_img1 = resample(x2, flownets1_flow)
        norm_diff_img0 = channel_norm(x1 - resampled_img1)

        # FlowNet-S
        concat2 = tf.concat([x1x2, resampled_img1, flownets1_flow / DISP_SCALE, norm_diff_img0], axis=1)
        with tf.variable_scope('flownet_s2'):
            flownets2_flow2 = FlowNet2S().graph_structure(concat2, standalone=False)
        flownets2_flow = resize(flownets2_flow2 * DISP_SCALE, mode='nearest')

        norm_flownets2_flow = channel_norm(flownets2_flow)
        diff_flownets2_flow = resample(x2, flownets2_flow)
        diff_flownets2_img1 = channel_norm(x1 - diff_flownets2_flow)

        # FlowNet-SD
        with tf.variable_scope('flownet_sd'):
            flownetsd_flow = self.flownet2_sd(x1x2)

        norm_flownetsd_flow = channel_norm(flownetsd_flow)
        diff_flownetsd_flow = resample(x2, flownetsd_flow)
        diff_flownetsd_img1 = channel_norm(x1 - diff_flownetsd_flow)

        concat3 = tf.concat([x1,
                            flownetsd_flow, flownets2_flow,
                            norm_flownetsd_flow, norm_flownets2_flow,
                            diff_flownetsd_img1, diff_flownets2_img1], axis=1)

        # FlowNet-Fusion
        with tf.variable_scope('flownet_fusion'):
            flownetfusion_flow = self.flownet2_fusion(concat3)

        return flownetfusion_flow

    def flownet2_fusion(self, x):
        """
        Architecture in Table 4 of FlowNet 2.0.

        Args:
            x: NCHW tensor, where C=11 is the concatenation of 7 items of [3, 2, 2, 1, 1, 1, 1] channels.
        """
        with argscope([tf.layers.conv2d], activation=lambda x: tf.nn.leaky_relu(x, 0.1),
                      padding='valid', strides=2, kernel_size=3,
                      data_format='channels_first'), \
            argscope([tf.layers.conv2d_transpose], padding='same', activation=tf.identity,
                     data_format='channels_first', strides=2, kernel_size=4):
            conv0 = tf.layers.conv2d(pad(x, 1), 64, name='conv0', strides=1)

            x = tf.layers.conv2d(pad(conv0, 1), 64, name='conv1')
            conv1 = tf.layers.conv2d(pad(x, 1), 128, name='conv1_1', strides=1)
            x = tf.layers.conv2d(pad(conv1, 1), 128, name='conv2')
            conv2 = tf.layers.conv2d(pad(x, 1), 128, name='conv2_1', strides=1)

            flow2 = tf.layers.conv2d(pad(conv2, 1), 2, name='predict_flow2', strides=1, activation=tf.identity)
            flow2_up = tf.layers.conv2d_transpose(flow2, 2, name='upsampled_flow2_to_1')
            x = tf.layers.conv2d_transpose(conv2, 32, name='deconv1', activation=lambda x: tf.nn.leaky_relu(x, 0.1))

            concat1 = tf.concat([conv1, x, flow2_up], axis=1, name='concat1')
            interconv1 = tf.layers.conv2d(pad(concat1, 1), 32, strides=1, name='inter_conv1', activation=tf.identity)

            flow1 = tf.layers.conv2d(pad(interconv1, 1), 2, name='predict_flow1', strides=1, activation=tf.identity)
            flow1_up = tf.layers.conv2d_transpose(flow1, 2, name='upsampled_flow1_to_0')
            x = tf.layers.conv2d_transpose(concat1, 16, name='deconv0', activation=lambda x: tf.nn.leaky_relu(x, 0.1))

            concat0 = tf.concat([conv0, x, flow1_up], axis=1, name='concat0')
            interconv0 = tf.layers.conv2d(pad(concat0, 1), 16, strides=1, name='inter_conv0', activation=tf.identity)
            flow0 = tf.layers.conv2d(pad(interconv0, 1), 2, name='predict_flow0', strides=1, activation=tf.identity)

            return tf.identity(flow0, name='flow2')

    def flownet2_sd(self, x):
        """
        Architecture in Table 3 of FlowNet 2.0.

        Args:
            x: concatenation of two inputs, of shape [1, 2xC, H, W]
        """
        with argscope([tf.layers.conv2d], activation=lambda x: tf.nn.leaky_relu(x, 0.1),
                      padding='valid', strides=2, kernel_size=3,
                      data_format='channels_first'), \
            argscope([tf.layers.conv2d_transpose], padding='same', activation=tf.identity,
                     data_format='channels_first', strides=2, kernel_size=4):
            x = tf.layers.conv2d(pad(x, 1), 64, name='conv0', strides=1)

            x = tf.layers.conv2d(pad(x, 1), 64, name='conv1')
            conv1 = tf.layers.conv2d(pad(x, 1), 128, name='conv1_1', strides=1)
            x = tf.layers.conv2d(pad(conv1, 1), 128, name='conv2')
            conv2 = tf.layers.conv2d(pad(x, 1), 128, name='conv2_1', strides=1)

            x = tf.layers.conv2d(pad(conv2, 1), 256, name='conv3')
            conv3 = tf.layers.conv2d(pad(x, 1), 256, name='conv3_1', strides=1)
            x = tf.layers.conv2d(pad(conv3, 1), 512, name='conv4')
            conv4 = tf.layers.conv2d(pad(x, 1), 512, name='conv4_1', strides=1)
            x = tf.layers.conv2d(pad(conv4, 1), 512, name='conv5')
            conv5 = tf.layers.conv2d(pad(x, 1), 512, name='conv5_1', strides=1)
            x = tf.layers.conv2d(pad(conv5, 1), 1024, name='conv6')
            conv6 = tf.layers.conv2d(pad(x, 1), 1024, name='conv6_1', strides=1)

            flow6 = tf.layers.conv2d(pad(conv6, 1), 2, name='predict_flow6', strides=1, activation=tf.identity)
            flow6_up = tf.layers.conv2d_transpose(flow6, 2, name='upsampled_flow6_to_5')
            x = tf.layers.conv2d_transpose(conv6, 512, name='deconv5', activation=lambda x: tf.nn.leaky_relu(x, 0.1))

            concat5 = tf.concat([conv5, x, flow6_up], axis=1, name='concat5')
            interconv5 = tf.layers.conv2d(pad(concat5, 1), 512, strides=1, name='inter_conv5', activation=tf.identity)
            flow5 = tf.layers.conv2d(pad(interconv5, 1), 2, name='predict_flow5', strides=1, activation=tf.identity)
            flow5_up = tf.layers.conv2d_transpose(flow5, 2, name='upsampled_flow5_to_4')
            x = tf.layers.conv2d_transpose(concat5, 256, name='deconv4', activation=lambda x: tf.nn.leaky_relu(x, 0.1))

            concat4 = tf.concat([conv4, x, flow5_up], axis=1, name='concat4')
            interconv4 = tf.layers.conv2d(pad(concat4, 1), 256, strides=1, name='inter_conv4', activation=tf.identity)
            flow4 = tf.layers.conv2d(pad(interconv4, 1), 2, name='predict_flow4', strides=1, activation=tf.identity)
            flow4_up = tf.layers.conv2d_transpose(flow4, 2, name='upsampled_flow4_to_3')
            x = tf.layers.conv2d_transpose(concat4, 128, name='deconv3', activation=lambda x: tf.nn.leaky_relu(x, 0.1))

            concat3 = tf.concat([conv3, x, flow4_up], axis=1, name='concat3')
            interconv3 = tf.layers.conv2d(pad(concat3, 1), 128, strides=1, name='inter_conv3', activation=tf.identity)
            flow3 = tf.layers.conv2d(pad(interconv3, 1), 2, name='predict_flow3', strides=1, activation=tf.identity)
            flow3_up = tf.layers.conv2d_transpose(flow3, 2, name='upsampled_flow3_to_2')
            x = tf.layers.conv2d_transpose(concat3, 64, name='deconv2', activation=lambda x: tf.nn.leaky_relu(x, 0.1))

            concat2 = tf.concat([conv2, x, flow3_up], axis=1, name='concat2')
            interconv2 = tf.layers.conv2d(pad(concat2, 1), 64, strides=1, name='inter_conv2', activation=tf.identity)
            flow2 = tf.layers.conv2d(pad(interconv2, 1), 2, name='predict_flow2', strides=1, activation=tf.identity)

            return resize(flow2 / DISP_SCALE, mode='nearest')


class FlowNet2S(FlowNetBase):
    def graph_structure(self, x, standalone=True):
        """
        Architecture of FlowNetSimple in Figure 2 of FlowNet 1.0.

        Args:
            x: 2CHW if standalone==True, else NCHW where C=12 is a concatenation
                of 5 tensors of [3, 3, 3, 2, 1] channels.
            standalone: If True, this model is used to predict flow from two inputs.
                If False, this model is used as part of the FlowNet2.
        """
        if standalone:
            x = tf.concat(tf.split(x, 2, axis=0), axis=1)

        with argscope([tf.layers.conv2d], activation=lambda x: tf.nn.leaky_relu(x, 0.1),
                      padding='valid', strides=2, kernel_size=3,
                      data_format='channels_first'), \
            argscope([tf.layers.conv2d_transpose], padding='same', activation=tf.identity,
                     data_format='channels_first', strides=2, kernel_size=4):
            x = tf.layers.conv2d(pad(x, 3), 64, kernel_size=7, name='conv1')
            conv2 = tf.layers.conv2d(pad(x, 2), 128, kernel_size=5, name='conv2')
            x = tf.layers.conv2d(pad(conv2, 2), 256, kernel_size=5, name='conv3')
            conv3 = tf.layers.conv2d(pad(x, 1), 256, name='conv3_1', strides=1)
            x = tf.layers.conv2d(pad(conv3, 1), 512, name='conv4')
            conv4 = tf.layers.conv2d(pad(x, 1), 512, name='conv4_1', strides=1)
            x = tf.layers.conv2d(pad(conv4, 1), 512, name='conv5')
            conv5 = tf.layers.conv2d(pad(x, 1), 512, name='conv5_1', strides=1)
            x = tf.layers.conv2d(pad(conv5, 1), 1024, name='conv6')
            conv6 = tf.layers.conv2d(pad(x, 1), 1024, name='conv6_1', strides=1)

            flow6 = tf.layers.conv2d(pad(conv6, 1), 2, name='predict_flow6', strides=1, activation=tf.identity)
            flow6_up = tf.layers.conv2d_transpose(flow6, 2, name='upsampled_flow6_to_5', use_bias=False)
            x = tf.layers.conv2d_transpose(conv6, 512, name='deconv5', activation=lambda x: tf.nn.leaky_relu(x, 0.1))

            concat5 = tf.concat([conv5, x, flow6_up], axis=1, name='concat5')
            flow5 = tf.layers.conv2d(pad(concat5, 1), 2, name='predict_flow5', strides=1, activation=tf.identity)
            flow5_up = tf.layers.conv2d_transpose(flow5, 2, name='upsampled_flow5_to_4', use_bias=False)
            x = tf.layers.conv2d_transpose(concat5, 256, name='deconv4', activation=lambda x: tf.nn.leaky_relu(x, 0.1))

            concat4 = tf.concat([conv4, x, flow5_up], axis=1, name='concat4')
            flow4 = tf.layers.conv2d(pad(concat4, 1), 2, name='predict_flow4', strides=1, activation=tf.identity)
            flow4_up = tf.layers.conv2d_transpose(flow4, 2, name='upsampled_flow4_to_3', use_bias=False)
            x = tf.layers.conv2d_transpose(concat4, 128, name='deconv3', activation=lambda x: tf.nn.leaky_relu(x, 0.1))

            concat3 = tf.concat([conv3, x, flow4_up], axis=1, name='concat3')
            flow3 = tf.layers.conv2d(pad(concat3, 1), 2, name='predict_flow3', strides=1, activation=tf.identity)
            flow3_up = tf.layers.conv2d_transpose(flow3, 2, name='upsampled_flow3_to_2', use_bias=False)
            x = tf.layers.conv2d_transpose(concat3, 64, name='deconv2', activation=lambda x: tf.nn.leaky_relu(x, 0.1))

            concat2 = tf.concat([conv2, x, flow3_up], axis=1, name='concat2')
            flow2 = tf.layers.conv2d(pad(concat2, 1), 2, name='predict_flow2', strides=1, activation=tf.identity)

            return tf.identity(flow2, name='flow2')


class FlowNet2C(FlowNetBase):
    def graph_structure(self, x1x2):
        """
        Architecture of FlowNetCorr in Figure 2 of FlowNet 1.0.
        Args:
            x: 2CHW.
        """
        with argscope([tf.layers.conv2d], activation=lambda x: tf.nn.leaky_relu(x, 0.1),
                      padding='valid', strides=2, kernel_size=3,
                      data_format='channels_first'), \
            argscope([tf.layers.conv2d_transpose], padding='same', activation=tf.identity,
                     data_format='channels_first', strides=2, kernel_size=4):

            # extract features
            x = tf.layers.conv2d(pad(x1x2, 3), 64, kernel_size=7, name='conv1')
            conv2 = tf.layers.conv2d(pad(x, 2), 128, kernel_size=5, name='conv2')
            conv3 = tf.layers.conv2d(pad(conv2, 2), 256, kernel_size=5, name='conv3')

            conv2a, _ = tf.split(conv2, 2, axis=0)
            conv3a, conv3b = tf.split(conv3, 2, axis=0)

            corr = correlation(conv3a, conv3b,
                               kernel_size=1,
                               max_displacement=20,
                               stride_1=1,
                               stride_2=2,
                               pad=20, data_format='NCHW')
            corr = tf.nn.leaky_relu(corr, 0.1)

            conv_redir = tf.layers.conv2d(conv3a, 32, kernel_size=1, strides=1, name='conv_redir')

            in_conv3_1 = tf.concat([conv_redir, corr], axis=1, name='in_conv3_1')
            conv3_1 = tf.layers.conv2d(pad(in_conv3_1, 1), 256, name='conv3_1', strides=1)

            x = tf.layers.conv2d(pad(conv3_1, 1), 512, name='conv4')
            conv4 = tf.layers.conv2d(pad(x, 1), 512, name='conv4_1', strides=1)
            x = tf.layers.conv2d(pad(conv4, 1), 512, name='conv5')
            conv5 = tf.layers.conv2d(pad(x, 1), 512, name='conv5_1', strides=1)
            x = tf.layers.conv2d(pad(conv5, 1), 1024, name='conv6')
            conv6 = tf.layers.conv2d(pad(x, 1), 1024, name='conv6_1', strides=1)

            flow6 = tf.layers.conv2d(pad(conv6, 1), 2, name='predict_flow6', strides=1, activation=tf.identity)
            flow6_up = tf.layers.conv2d_transpose(flow6, 2, name='upsampled_flow6_to_5')
            x = tf.layers.conv2d_transpose(conv6, 512, name='deconv5', activation=lambda x: tf.nn.leaky_relu(x, 0.1))

            # return flow6
            concat5 = tf.concat([conv5, x, flow6_up], axis=1, name='concat5')
            flow5 = tf.layers.conv2d(pad(concat5, 1), 2, name='predict_flow5', strides=1, activation=tf.identity)
            flow5_up = tf.layers.conv2d_transpose(flow5, 2, name='upsampled_flow5_to_4')
            x = tf.layers.conv2d_transpose(concat5, 256, name='deconv4', activation=lambda x: tf.nn.leaky_relu(x, 0.1))

            concat4 = tf.concat([conv4, x, flow5_up], axis=1, name='concat4')
            flow4 = tf.layers.conv2d(pad(concat4, 1), 2, name='predict_flow4', strides=1, activation=tf.identity)
            flow4_up = tf.layers.conv2d_transpose(flow4, 2, name='upsampled_flow4_to_3')
            x = tf.layers.conv2d_transpose(concat4, 128, name='deconv3', activation=lambda x: tf.nn.leaky_relu(x, 0.1))

            concat3 = tf.concat([conv3_1, x, flow4_up], axis=1, name='concat3')
            flow3 = tf.layers.conv2d(pad(concat3, 1), 2, name='predict_flow3', strides=1, activation=tf.identity)
            flow3_up = tf.layers.conv2d_transpose(flow3, 2, name='upsampled_flow3_to_2')
            x = tf.layers.conv2d_transpose(concat3, 64, name='deconv2', activation=lambda x: tf.nn.leaky_relu(x, 0.1))

            concat2 = tf.concat([conv2a, x, flow3_up], axis=1, name='concat2')
            flow2 = tf.layers.conv2d(pad(concat2, 1), 2, name='predict_flow2', strides=1, activation=tf.identity)

            return tf.identity(flow2, name='flow2')
