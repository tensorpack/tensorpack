
# Trainers

Tensorpack trainers contain logic of:

1. Building the graph.
2. Running the iterations (with callbacks).

Usually you won't touch these methods directly, but use
[higher-level interface](training-interface.html) on trainers.
You'll only need to __select__ what trainer to use.

### Tower Trainer

Following the terminology in TensorFlow,
a __tower function__ is a callable that takes input tensors and adds __one replicate__ of the model to the graph.

Most types of neural-network training could fall into this category.
All non-base trainers in tensorpack is a subclass of [TowerTrainer](../modules/train.html#tensorpack.train.TowerTrainer).
The concept of tower is used mainly to support:

1. Data-parallel multi-GPU training, where a replicate is built on each GPU.
2. Automatically building the graph for inference, where a replicate is built under inference mode.

You'll specify a tower function when you use `TowerTrainer`.
The function needs to follow some conventions:

1. It will always be called under a :class:`TowerContext`.
	 which will contain information about reuse, training/inference, scope name, etc.
2. It might get called multiple times for data-parallel training or inference.
3. To respect variable reuse, use `tf.get_variable` instead of
	 `tf.Variable` in the function.


### MultiGPU Trainers

For data-parallel multi-GPU training, different [multi-GPU trainers](http://tensorpack.readthedocs.io/en/latest/modules/train.html)
implement different parallel logic.
They take care of device placement, gradient averaging and synchronoization
in the efficient way and all reach the same performance as the
[official TF benchmarks](https://www.tensorflow.org/performance/benchmarks).
It takes only one line of code change to use them.

Note some common problems when using these trainers:

1. In each iteration all GPUs (all replicates of the model) will take tensors from the `InputSource`,
	instead of taking one for all and split.
	So the total batch size would become ``(batch size of InputSource/DataFlow) * #GPU``.

	Splitting a tensor to GPUs makes no sense at all, only to put unnecessary shape constraints on the data.
	By letting each GPU train on its own input tensors, they can train on inputs of different shapes simultaneously.

2. The tower function (your model code) will get called multipile times.
	You'll need to be very careful when modifying global states in those functions, e.g. adding ops to TF collections.
