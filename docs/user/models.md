
# Model

To define a model (i.e. the computation graph) that will be used for training,
you'll need to subclass `ModelDesc` and implement several methods:

```python
class MyModel(ModelDesc):
	def _get_inputs(self):
		return [InputVar(...), InputVar(...)]

	def _build_graph(self, inputs):
		# build the graph
```

Basically, `_get_inputs` should define the metainfo of the input
of the model. It should match what is produced by the data you're training with.
`_build_graph` should add tensors/operations to the graph, where
the argument `input_tensors` is the list of input tensors matching the return value of
`_get_inputs`.

You can use any symbolic functions in `_build_graph`, including TensorFlow core library
functions, TensorFlow slim layers, or functions in other packages such as tflean, tensorlayer.

tensorpack also contains a small collection of common model primitives,
such as conv/deconv, fc, pooling layers, nonlinearities, and some custom loss functions.
Using the tensorpack implementations, you can also benefit from `argscope` and `LinearWrap` to
simplify the code.

## argscope and LinearWrap
`argscope` gives you a context with default arguments.
`LinearWrap` allows you to simplify "linear structure" models by
adding the layers one by one.

The following code:
```python
with argscope(Conv2D, out_channel=32, kernel_shape=3, nl=tf.nn.relu):
	l = (LinearWrap(image)  # the starting brace is only for line-breaking
			 .Conv2D('conv0')
			 .MaxPooling('pool0', 2)
			 .Conv2D('conv1', padding='SAME')
			 .Conv2D('conv2', kernel_shape=5)
			 .FullyConnected('fc0', 512, nl=tf.nn.relu)
			 .Dropout('dropout', 0.5)
			 .FullyConnected('fc1', out_dim=10, nl=tf.identity)())
```
is equivalent to:
```
l = Conv2D('conv0', image, 32, 3, nl=tf.nn.relu)
l = MaxPooling('pool0', l, 2)
l = Conv2D('conv1', l, 32, 3, padding='SAME')
l = Conv2D('conv2', l, 32, 5)
l = FullyConnected('fc0', l, 512, nl=tf.nn.relu)
l = Dropout('dropout', l, 0.5)
l = FullyConnected('fc1', l, 10, nl=tf.identity)
```

## Implement a layer

Symbolic functions should be nothing new to you, and writing a simple symbolic function is nothing special in tensorpack.
But you can make a symbolic function become a "layer" by following some very simple rules, and then gain benefits from the framework.

Take a look at the [Convolutional Layer](../tensorpack/models/conv2d.py#L14) implementation for an example of how to define a
model primitive:

```python
@layer_register()
def Conv2D(x, out_channel, kernel_shape,
           padding='SAME', stride=1,
           W_init=None, b_init=None,
           nl=tf.nn.relu, split=1, use_bias=True):
```

Basically, a layer is a symbolic function with the following rules:

+ It is decorated by `@layer_register`.
+ The first argument is its "input". It must be a tensor or a list of tensors.
+ It returns either a tensor or a list of tensors as its "output".


By making a symbolic function a "layer", the following thing will happen:
+ You will call the function with a scope argument, e.g. `Conv2D('conv0', x, 32, 3)`.
	Everything happening in this function will be under the variable scope 'conv0'. You can register
	the layer with `use_scope=False` to disable this feature.
+ Static shapes of input/output will be logged.
+ It will then work with `argscope` to easily define default arguments. `argscope` will work for all
	the arguments except the input.
+ It will work with `LinearWrap` if the output of the previous layer matches the input of the next layer.

Take a look at the [Inception example](../examples/Inception/inception-bn.py#L36) to see how a complicated model can be described with these primitives.

There are also a number of symbolic functions in the `tfutils.symbolic_functions` module.
There isn't a rule about what kind of symbolic functions should be made a layer -- they're quite
similar anyway. But in general I define the following kinds of symbolic functions as layers:
+ Functions which contain variables. A variable scope is almost always helpful for such function.
+ Functions which are commonly referred to as "layers", such as pooling. This make a model
	definition more straightforward.

