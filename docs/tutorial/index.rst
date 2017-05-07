
Tutorials
---------------------

A High Level Glance
====================

* :doc:`dataflow` is a set of extensible tools to help you define your input data with ease and speed.

  It provides a uniform interface so that data processing modules can be chained together.
  It allows you to load and process your data in pure Python and accelerate it by prefetching.
  See also :doc:`input-source`  and :doc:`efficient-dataflow` for more details about the efficiency of data
  processing.

* You can use any TF-based symbolic function library to define a model in tensorpack.
  :doc:`model` introduces where and how you define the model for tensorpack trainers to use,
  and how you can benefit from the small symbolic function library in tensorpack.

Both DataFlow and models can be used outside tensorpack, as just a data processing library and a symbolic
function library. Tensopack trainers integrate these two components and add more convenient features.

* tensorpack :doc:`trainer` manages the training loops for you, so you will not have to worry about
  details such as multi-GPU training. At the same time, it keeps the power of customization
  through callbacks.

* Callbacks are like ``tf.train.SessionRunHook``, or plugins, or extensions. During training,
  everything you want to do other than the main iterations can be defined through callbacks.
  See :doc:`callback` for some examples what you can do.

User Tutorials
========================

.. toctree::
  :maxdepth: 1

  dataflow
  input-source
  efficient-dataflow
  model
  trainer
  callback
  faq

Extend Tensorpack
=================

.. toctree::
  :maxdepth: 1

  extend/dataflow
  extend/augmentor
  extend/model
  extend/trainer
  extend/callback
