
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

Most tensorpack trainers expect a `ModelDesc`.
The trainers will call these methods to create the model,
connect `InputSource` to the model, create the minimization op, and so on.
Data-parallel Multi-GPU trainers will call `_build_graph` __multiple times__ on each GPU.
A trainer may also make __extra calls__ to `_build_graph` for inference, if used by some callbacks.

### Build It Manually

When you need to deal with complicated graph, it may be easier to build the graph manually.
You are free to do so as long as you tell the trainer what to do in each step.

More details to come.
