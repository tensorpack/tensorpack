
# Trainers

Tensorpack trainers contain logic of:

1. Building the graph.
2. Running the iterations (with callbacks).

Usually you won't touch these methods directly, but use
[higher-level interface](training-interface.html) on trainers.
You'll only need to __select__ what trainer to use.
But some basic knowledge of how they work is useful:

### Tower Trainer

Following the terminology in TensorFlow,
a __tower function__ is a callable that takes input tensors and adds __one replicate__ of the model to the graph.

Most types of neural-network training could fall into this category.
All trainers in tensorpack is a subclass of [TowerTrainer](../modules/train.html#tensorpack.train.TowerTrainer).
The concept of tower is used mainly to support:

1. Data-parallel multi-GPU training, where a replicate is built on each GPU.
2. Graph construction for inference, where a replicate is built under inference mode.

You'll provide a tower function to use `TowerTrainer`.
The function needs to follow some conventions:

1. It will always be called under a `TowerContext`.
	 which will contain information about reuse, training/inference, scope name, etc.
2. __It might get called multiple times__ for data-parallel training or inference.
3. To respect variable reuse, use `tf.get_variable` instead of
	 `tf.Variable` in the function, unless you want to force creation of new variables.

In particular, when working with the `ModelDesc` interface, its `build_graph` method will be the tower function.

### MultiGPU Trainers

For data-parallel multi-GPU training, different [multi-GPU trainers](../modules/train.html)
implement different parallel logic.
They take care of device placement, gradient averaging and synchronoization
in the efficient way and all reach the same performance as the
[official TF benchmarks](https://www.tensorflow.org/performance/benchmarks).
It takes only one line of code change to use them.

Note some __common problems__ when using these trainers:

1. In each iteration, all GPUs (all replicates of the model) take tensors from the `InputSource`,
	instead of take one for all and split.
	So the total batch size would become ``(batch size of InputSource/DataFlow) * #GPU``.

	Splitting a tensor for data-parallel training makes no sense at all, only to put unnecessary shape constraints on the data.
	By letting each GPU train on its own input tensors, they can train on inputs of different shapes simultaneously.

2. The tower function (your model code) will get called multipile times.
	You'll need to be very careful when modifying global states in those functions, e.g. adding ops to TF collections.
