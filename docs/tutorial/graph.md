
# Build the Graph

### ModelDesc

`ModelDesc` is an abstraction over the most common type of models people train.
It assumes:

1. Training is a single-cost optimized by a single `tf.train.Optimizer`.
2. The graph can be trivially duplicated for data-parallel training or inference.

If your task is single-cost optimization,
you can subclass `ModelDesc` and implement several methods:

```python
class MyModel(ModelDesc):
	def _get_inputs(self):
		return [InputDesc(...), InputDesc(...)]

	def _build_graph(self, inputs):
		tensorA, tensorB = inputs
		# build the graph
		self.cost = xxx	 # define the cost tensor

	def _get_optimizer(self):
	  return tf.train.GradientDescentOptimizer(0.1)
```

`_get_inputs` should define the metainfo of all the inputs your graph may need.
`_build_graph` should add tensors/operations to the graph, where
the argument `inputs` is the list of input tensors matching `_get_inputs`.
You can use any symbolic functions in `_build_graph`, including TensorFlow core library
functions and other symbolic libraries.

### How it is Used:

Most tensorpack trainers expect a `ModelDesc`, and use it as a __description
of the TF graph to be built__.
These trainers will use `_get_inputs` to connect the given `InputSource` to the graph.
They'll then use `_build_graph` to create the backbone model, and then `_get_optimizer` to create the minimization op, and run it.

Note that data-parallel multi-GPU trainers will call `_build_graph` __multiple times__ on each GPU.
A trainer may also make __extra calls__ to `_build_graph` for inference, if used by some callbacks.
`_build_graph` will always be called under some `TowerContext` which contains these context information
(e.g. training or inference, reuse or not, scope name) for your access.

Also, to respect variable reuse among multiple calls, use `tf.get_variable()` instead of `tf.Variable` in `_build_graph`,
if you need to create any variables.

### Build It Manually

When you need to deal with complicated graph, it may be easier to build the graph manually.
You are free to do so as long as you tell the trainer what to do in each step.

Check out [Write a Trainer](extend/trainer.html)
for using a custom graph with trainer.
