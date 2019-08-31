# -*- coding: utf-8 -*-
# File: DQNModel.py


import abc
import tensorflow as tf

from tensorpack import ModelDesc
from tensorpack.tfutils import gradproc, optimizer, summary, varreplace
from tensorpack.tfutils.scope_utils import auto_reuse_variable_scope
from tensorpack.utils import logger


class Model(ModelDesc):

    state_dtype = tf.uint8

    # reward discount factor
    gamma = 0.99

    def __init__(self, state_shape, history, method, num_actions):
        """
        Args:
            state_shape (tuple[int]),
            history (int):
        """
        self.state_shape = tuple(state_shape)
        self._stacked_state_shape = (-1, ) + self.state_shape + (history, )
        self.history = history
        self.method = method
        self.num_actions = num_actions

    def inputs(self):
        # When we use h history frames, the current state and the next state will have (h-1) overlapping frames.
        # Therefore we use a combined state for efficiency:
        # The first h are the current state, and the last h are the next state.
        return [tf.TensorSpec((None,) + self.state_shape + (self.history + 1, ), self.state_dtype, 'comb_state'),
                tf.TensorSpec((None,), tf.int64, 'action'),
                tf.TensorSpec((None,), tf.float32, 'reward'),
                tf.TensorSpec((None,), tf.bool, 'isOver')]

    @abc.abstractmethod
    def _get_DQN_prediction(self, state):
        """
        state: N + state_shape + history
        """
        pass

    @auto_reuse_variable_scope
    def get_DQN_prediction(self, state):
        return self._get_DQN_prediction(state)

    def build_graph(self, comb_state, action, reward, isOver):
        comb_state = tf.cast(comb_state, tf.float32)
        input_rank = comb_state.shape.rank

        state = tf.slice(
            comb_state,
            [0] * input_rank,
            [-1] * (input_rank - 1) + [self.history], name='state')

        self.predict_value = self.get_DQN_prediction(state)
        if not self.training:
            return

        reward = tf.clip_by_value(reward, -1, 1)
        next_state = tf.slice(
            comb_state,
            [0] * (input_rank - 1) + [1],
            [-1] * (input_rank - 1) + [self.history], name='next_state')
        next_state = tf.reshape(next_state, self._stacked_state_shape)
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
        lr = tf.get_variable('learning_rate', initializer=1e-3, trainable=False)
        tf.summary.scalar("learning_rate-summary", lr)
        opt = tf.train.RMSPropOptimizer(lr, decay=0.95, momentum=0.95, epsilon=1e-2)
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
