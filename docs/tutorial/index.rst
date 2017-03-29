
Tutorials
---------------------

A High Level Glance
====================

* :doc:`dataflow` is a set of extensible tools to help you define your input data with ease and speed.

  It provides a uniformed interface so data processing modules can be chained together.
  It allows you to load and process your data in pure Python and accelerate it by multiprocess prefetch.
  See also :doc:`tf-queue`  and :doc:`efficient-dataflow` for more details about efficiency of data
  processing.

* You can use any TF-based symbolic function library to define a model in tensorpack.
  :doc:`model` introduces where and how you define the model for tensorpack trainers to use,
  and how you can benefit from the symbolic function library in tensorpack.

Both DataFlow and models can be used outside tensorpack, as just a data processing library and a symbolic
function library. Tensopack trainers integrate these two components and add more convenient features.

* tensorpack :doc:`trainer` manages the training loops for you so you won't have to worry about
  details such as multi-GPU training. At the same time it keeps the power of customization to you
  through callbacks.

* Callbacks are like ``tf.train.SessionRunHook``, or plugins, or extensions. During training,
  everything you want to do other than the main iterations can be defined through callbacks.
  See :doc:`callback` for some examples what you can do.

User Tutorials
========================

.. toctree::
  :maxdepth: 1

  dataflow
  tf-queue
  efficient-dataflow
  model
  trainer
  callback

Extend Tensorpack
=================

.. toctree::
  :maxdepth: 1

  extend/dataflow
  extend/model
  extend/trainer
  extend/callback
