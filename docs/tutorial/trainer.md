
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
Most types of neural-network training could be described with this concept.
The concept of tower is used mainly to support:

1. Data-parallel multi-GPU training, where a replicate is built on each GPU.
2. Graph construction for inference, where a replicate is built under inference mode.

A user needs to provide a tower function to use `TowerTrainer`.
In particular, when working with the `ModelDesc` interface, the `build_graph` method will be the tower function.

The tower function needs to follow some conventions:

1. __It might get called multiple times__ for data-parallel training or inference.
2. It has to respect variable collections:
   * Only put variables __trainable by gradient descent__ into `TRAINABLE_VARIABLES`.
   * Put variables that need to be saved into `MODEL_VARIABLES`.
3. It has to respect variable scopes:
   * The name of any trainable variables created in the function must be like "variable_scope_name/variable/name".
     Don't depend on name_scope's name. Don't use variable_scope's name twice.
   * The creation of any trainable variables must respect variable reuse.
     To respect variable reuse, use `tf.get_variable` instead of
	 `tf.Variable` in the function.
     For non-trainable variables, it's OK to use `tf.Variable` to force creation of new variables in each tower.
4. It will always be called under a `TowerContext`.
	which will contain information about training/inference mode, reuse, etc.
     
These conventions are easy to follow, and most layer wrappers (e.g.,
tf.layers/slim/tensorlayer) do follow them. Note that certain Keras layers do not
follow these conventions and may crash if used within tensorpack.

It's possible to write ones that are not, but all existing trainers in
tensorpack are subclass of [TowerTrainer](../modules/train.html#tensorpack.train.TowerTrainer).

### MultiGPU Trainers

For data-parallel multi-GPU training, different [multi-GPU trainers](../modules/train.html)
implement different distribution strategies.
They take care of device placement, gradient averaging and synchronoization
in the efficient way and all reach the same performance as the
[official TF benchmarks](https://www.tensorflow.org/performance/benchmarks).
It takes only one line of code change to use them.

Note some __common problems__ when using these trainers:

1. In each iteration, all GPUs (all replicates of the model) take tensors from the `InputSource`,
	instead of taking one for all and split.
	So the total batch size would become ``(batch size of InputSource) * #GPU``.

	Splitting a tensor for data-parallel training makes no sense at all, only to put unnecessary shape constraints on the data.
	By letting each GPU train on its own input tensors, they can train on inputs of different shapes simultaneously.

2. The tower function (your model code) will get called multipile times.
	As a result, you'll need to be careful when modifying global states in those functions, e.g. adding ops to TF collections.
