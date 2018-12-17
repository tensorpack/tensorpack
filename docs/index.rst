Tensorpack Documentation
==============================

.. image:: ../.github/tensorpack.png

Tensorpack is a **training interface** based on TensorFlow, with a focus on speed + flexibility.
TensorFlow is powerful, but has its own drawbacks:
Its low-level APIs are too hard and complicated for many users,
and its existing high-level APIs sacrifice a lot in either speed or flexibility.
The Tensorpack API brings speed and flexibility together.

Tensorpack is Yet Another TF high-level API, but different in:

- Focus on **training speed**.

  - Speed comes for free with tensorpack -- it uses TensorFlow in the
    **efficient way** with no extra overhead. On common CNNs, it runs 
    `1.2~5x faster <https://github.com/tensorpack/benchmarks/tree/master/other-wrappers>`_
    than the equivalent Keras code.

  - Data-parallel multi-GPU/distributed training strategy is off-the-shelf to use. 
    It scales as well as Google's
    `official benchmark <https://www.tensorflow.org/performance/benchmarks>`_.
    You cannot beat its speed unless you're a TensorFlow expert.

  - See `tensorpack/benchmarks <https://github.com/tensorpack/benchmarks>`_ for some benchmark scripts.

- Focus on **large datasets**.

  - You don't usually need `tf.data`. Symbolic programming often makes data processing harder.
    Tensorpack helps you efficiently process large datasets (e.g. ImageNet) in **pure Python** with autoparallelization.

- It's not a model wrapper.

  - There are already too many symbolic function wrappers in the world.
    Tensorpack includes only a few common models, but you can use any symbolic function library inside tensorpack, including tf.layers/Keras/slim/tflearn/tensorlayer/...

See :doc:`tutorial/index` to know more about these features:


.. toctree::
  :maxdepth: 3

  tutorial/index
  modules/index
