# -*- coding: utf-8 -*-
# File: DQNModel.py


import abc
import tensorflow as tf

from tensorpack import ModelDesc
from tensorpack.tfutils import get_current_tower_context, gradproc, optimizer, summary, varreplace
from tensorpack.tfutils.scope_utils import auto_reuse_variable_scope
from tensorpack.utils import logger


class Model(ModelDesc):
    learning_rate = 1e-3

    def __init__(self, image_shape, channel, history, method, num_actions, gamma):
        assert len(image_shape) == 2, image_shape

        self.channel = channel
        self._shape2d = tuple(image_shape)
        self._shape3d = self._shape2d + (channel, )
        self._shape4d_for_prediction = (-1, ) + self._shape2d + (history * channel, )
        self._channel = channel
        self.history = history
        self.method = method
        self.num_actions = num_actions
        self.gamma = gamma

    def inputs(self):
        # When we use h history frames, the current state and the next state will have (h-1) overlapping frames.
        # Therefore we use a combined state for efficiency:
        # The first h are the current state, and the last h are the next state.
        return [tf.placeholder(tf.uint8,
                               (None,) + self._shape2d +
                               ((self.history + 1) * self.channel,),
                               'comb_state'),
                tf.placeholder(tf.int64, (None,), 'action'),
                tf.placeholder(tf.float32, (None,), 'reward'),
                tf.placeholder(tf.bool, (None,), 'isOver')]

    @abc.abstractmethod
    def _get_DQN_prediction(self, image):
        pass

    @auto_reuse_variable_scope
    def get_DQN_prediction(self, image):
        """ image: [N, H, W, history * C] in [0,255]"""
        return self._get_DQN_prediction(image)

    def build_graph(self, comb_state, action, reward, isOver):
        comb_state = tf.cast(comb_state, tf.float32)
        comb_state = tf.reshape(
            comb_state, [-1] + list(self._shape2d) + [self.history + 1, self.channel])

        state = tf.slice(comb_state, [0, 0, 0, 0, 0], [-1, -1, -1, self.history, -1])
        state = tf.reshape(state, self._shape4d_for_prediction, name='state')
        self.predict_value = self.get_DQN_prediction(state)
        if not get_current_tower_context().is_training:
            return

        reward = tf.clip_by_value(reward, -1, 1)
        next_state = tf.slice(comb_state, [0, 0, 0, 1, 0], [-1, -1, -1, self.history, -1], name='next_state')
        next_state = tf.reshape(next_state, self._shape4d_for_prediction)
        action_onehot = tf.one_hot(action, self.num_actions, 1.0, 0.0)

        pred_action_value = tf.reduce_sum(self.predict_value * action_onehot, 1)  # N,
        max_pred_reward = tf.reduce_mean(tf.reduce_max(
            self.predict_value, 1), name='predict_reward')
        summary.add_moving_summary(max_pred_reward)

        with tf.variable_scope('target'), varreplace.freeze_variables(skip_collection=True):
            targetQ_predict_value = self.get_DQN_prediction(next_state)    # NxA

        if self.method != 'Double':
            # DQN
            best_v = tf.reduce_max(targetQ_predict_value, 1)    # N,
        else:
            # Double-DQN
            next_predict_value = self.get_DQN_prediction(next_state)
            self.greedy_choice = tf.argmax(next_predict_value, 1)   # N,
            predict_onehot = tf.one_hot(self.greedy_choice, self.num_actions, 1.0, 0.0)
            best_v = tf.reduce_sum(targetQ_predict_value * predict_onehot, 1)

        target = reward + (1.0 - tf.cast(isOver, tf.float32)) * self.gamma * tf.stop_gradient(best_v)

        cost = tf.losses.huber_loss(
            target, pred_action_value, reduction=tf.losses.Reduction.MEAN)
        summary.add_param_summary(('conv.*/W', ['histogram', 'rms']),
                                  ('fc.*/W', ['histogram', 'rms']))   # monitor all W
        summary.add_moving_summary(cost)
        return cost

    def optimizer(self):
        lr = tf.get_variable('learning_rate', initializer=self.learning_rate, trainable=False)
        opt = tf.train.RMSPropOptimizer(lr, epsilon=1e-5)
        return optimizer.apply_grad_processors(opt, [gradproc.SummaryGradient()])

    @staticmethod
    def update_target_param():
        vars = tf.global_variables()
        ops = []
        G = tf.get_default_graph()
        for v in vars:
            target_name = v.op.name
            if target_name.startswith('target'):
                new_name = target_name.replace('target/', '')
                logger.info("Target Network Update: {} <- {}".format(target_name, new_name))
                ops.append(v.assign(G.get_tensor_by_name(new_name + ':0')))
        return tf.group(*ops, name='update_target_network')
