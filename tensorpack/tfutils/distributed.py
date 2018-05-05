# -*- coding: utf-8 -*-
# File: distributed.py


import tensorflow as tf


def get_distributed_session_creator(server):
    """
    Args:
       server (tf.train.Server):

    Returns:
        tf.train.SessionCreator
    """

    server_def = server.server_def
    is_chief = (server_def.job_name == 'worker') and (server_def.task_index == 0)

    init_op = tf.global_variables_initializer()
    local_init_op = tf.local_variables_initializer()
    ready_op = tf.report_uninitialized_variables()
    ready_for_local_init_op = tf.report_uninitialized_variables(tf.global_variables())
    sm = tf.train.SessionManager(
        local_init_op=local_init_op,
        ready_op=ready_op,
        ready_for_local_init_op=ready_for_local_init_op,
        graph=tf.get_default_graph())

    # to debug wrong variable collection
    # from pprint import pprint
    # print("GLOBAL:")
    # pprint([(k.name, k.device) for k in tf.global_variables()])
    # print("LOCAL:")
    # pprint([(k.name, k.device) for k in tf.local_variables()])

    class _Creator(tf.train.SessionCreator):
        def create_session(self):
            if is_chief:
                return sm.prepare_session(master=server.target, init_op=init_op)
            else:
                tf.logging.set_verbosity(tf.logging.INFO)   # print message about uninitialized vars
                ret = sm.wait_for_session(master=server.target)
                tf.logging.set_verbosity(tf.logging.WARN)
                return ret

    return _Creator()
