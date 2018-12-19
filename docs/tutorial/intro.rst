
What is tensorpack?
~~~~~~~~~~~~~~~~~~~

Tensorpack is a **training interface** based on TensorFlow, which means:
you'll use mostly tensorpack high-level APIs to do training, rather than TensorFlow low-level APIs.

Why tensorpack?
~~~~~~~~~~~~~~~~~~~

TensorFlow is powerful, but has its own drawbacks:
Its low-level APIs are too hard and complicated for many users,
and its existing high-level APIs sacrifice a lot in either speed or flexibility.
The Tensorpack API brings speed and flexibility together.


Is TensorFlow Slow?
~~~~~~~~~~~~~~~~~~~~~

No it's not, but it's not easy to write it in an efficient way.

When **speed** is a concern, users will have to worry a lot about things unrelated to the model.
Code written with low-level APIs or other existing high-level wrappers is often suboptimal in speed.
Even most of the official TensorFlow examples are written for simplicity rather than efficiency,
which as a result makes people think TensorFlow is *slow*.

The `official TensorFlow benchmark <https://github.com/tensorflow/benchmarks/tree/master/scripts/tf_cnn_benchmarks>`_ said this in their README:

  These models are designed for performance. For models that have clean and easy-to-read implementations, see the TensorFlow Official Models.

which seems to suggest that you cannot have performance and ease-of-use together.
However you can have them both in tensorpack.
Tensorpack uses TensorFlow efficiently, and hides performance details under its APIs.
You no longer need to write
data prefetch, multi-GPU replication, device placement, variables synchronization -- anything that's unrelated to the model itself.
You still need to understand graph and learn to write models with TF, but performance is all taken care of by tensorpack.

A High Level Glance
~~~~~~~~~~~~~~~~~~~

.. image:: https://user-images.githubusercontent.com/1381301/29187907-2caaa740-7dc6-11e7-8220-e20ca52c3ca6.png


* ``DataFlow`` is a library to load data efficiently in Python.
  Apart from DataFlow, native TF operators can be used for data loading as well.
  They will eventually be wrapped under the same ``InputSource`` interface and go through prefetching.

* You can use any TF-based symbolic function library to define a model, including
  a small set of functions within tensorpack. ``ModelDesc`` is an interface to connect the model with the
  ``InputSource`` interface.

* Tensorpack trainers manage the training loops for you.
  They also include data parallel logic for multi-GPU or distributed training.
  At the same time, you have the power of customization through callbacks.

* Callbacks are like ``tf.train.SessionRunHook``, or plugins. During training,
  everything you want to do other than the main iterations can be defined through callbacks and easily reused.

* All the components, though work perfectly together, are highly decorrelated: you can:

  * Use DataFlow alone as a data loading library, without tensorfow at all.
  * Use tensorpack to build the graph with multi-GPU or distributed support,
    then train it with your own loops.
  * Build the graph on your own, and train it with tensorpack callbacks.
