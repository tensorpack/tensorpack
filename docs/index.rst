Tensorpack Documentation
==============================

.. image:: ../.github/tensorpack.png

Tensorpack is a **training interface** based on TensorFlow.

It's Yet Another TF wrapper, but different in:

- Focus on **training speed**.

  - Speed comes for free with tensorpack -- it uses TensorFlow in the
    **efficient way** with no extra overhead. On various CNNs, it runs 
    `1.2~5x faster <https://github.com/tensorpack/benchmarks/tree/master/other-wrappers>`_
    than the equivalent Keras code.

  - Data-parallel multi-GPU/distributed training strategy is off-the-shelf to use. 
    It is as fast as Google's
    `official benchmark <https://www.tensorflow.org/performance/benchmarks>`_.
    You cannot beat its speed unless you're a TensorFlow expert.

  - See `tensorpack/benchmarks <https://github.com/tensorpack/benchmarks>`_ for some benchmark scripts.

- Focus on **large datasets**.

  - It's unnecessary to read/preprocess data with a new language called TF.
    Tensorpack helps you load large datasets (e.g. ImageNet) in **pure Python** with autoparallelization.

- It's not a model wrapper.

  - There are already too many symbolic function wrappers in the world.
    Tensorpack includes only a few common models, but you can use any other wrappers within tensorpack, including sonnet/Keras/slim/tflearn/tensorlayer/....

See :doc:`tutorial/index` to know more about these features:


.. toctree::
  :maxdepth: 3

  tutorial/index
  modules/index
