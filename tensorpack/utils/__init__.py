#  -*- coding: UTF-8 -*-
#  File: __init__.py
#  Author: Yuxin Wu <ppwwyyxx@gmail.com>

from pkgutil import walk_packages
import os
import tensorflow as tf
import numpy as np

def global_import(name):
    p = __import__(name, globals(), None, level=1)
    lst = p.__all__ if '__all__' in dir(p) else dir(p)
    for k in lst:
        globals()[k] = p.__dict__[k]
global_import('naming')
#global_import('sessinit')
global_import('utils')

# TODO move this utils to another file
#def get_default_sess_config(mem_fraction=0.5):
    #"""
    #Return a better config to use as default.
    #Tensorflow default session config consume too much resources
    #"""
    #conf = tf.ConfigProto()
    #conf.gpu_options.per_process_gpu_memory_fraction = mem_fraction
    #conf.gpu_options.allocator_type = 'BFC'
    #conf.allow_soft_placement = True
    #return conf

#def get_global_step_var():
    #""" get global_step variable in the current graph"""
    #try:
        #return tf.get_default_graph().get_tensor_by_name(GLOBAL_STEP_VAR_NAME)
    #except KeyError:
        #var = tf.Variable(
            #0, trainable=False, name=GLOBAL_STEP_OP_NAME)
        #return var

#def get_global_step():
    #""" get global_step value with current graph and session"""
    #return tf.train.global_step(
        #tf.get_default_session(),
        #get_global_step_var())
