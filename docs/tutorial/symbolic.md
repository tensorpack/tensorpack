
# Symbolic Layers

While you can use other symbolic libraries,
tensorpack also contains a small collection of common model primitives,
such as conv/deconv, fc, bn, pooling layers.
Using the tensorpack implementations, you can also benefit from `argscope` and `LinearWrap` to
simplify the code.

Note that these layers were written because there were no other alternatives back at that time.
Now, these layers actually call `tf.layers` directly.
You can just use `tf.layers` as long as it fits your need.

### argscope and LinearWrap
`argscope` gives you a context with default arguments.
`LinearWrap` is a syntax sugar to simplify building "linear structure" models.

The following code:
```python
with argscope(Conv2D, filters=32, kernel_size=3, activation=tf.nn.relu):
  l = (LinearWrap(image)  # the starting brace is only for line-breaking
       .Conv2D('conv0')
       .MaxPooling('pool0', 2)
       .Conv2D('conv1', padding='SAME')
       .Conv2D('conv2', kernel_size=5)
       .FullyConnected('fc0', 512, activation=tf.nn.relu)
       .Dropout('dropout', rate=0.5)
       .tf.multiply(0.5)
       .apply(func, *args, **kwargs)
       .FullyConnected('fc1', units=10, activation=tf.identity)())
```
is equivalent to:
```
l = Conv2D('conv0', image, 32, 3, activation=tf.nn.relu)
l = MaxPooling('pool0', l, 2)
l = Conv2D('conv1', l, 32, 3, padding='SAME', activation=tf.nn.relu)
l = Conv2D('conv2', l, 32, 5, activation=tf.nn.relu)
l = FullyConnected('fc0', l, 512, activation=tf.nn.relu)
l = Dropout('dropout', l, rate=0.5)
l = tf.multiply(l, 0.5)
l = func(l, *args, **kwargs)
l = FullyConnected('fc1', l, 10, activation=tf.identity)
```

### Access Relevant Tensors

The variables inside the layer will be named `name/W`, `name/b`, etc.
See the API documentation of each layer for details.
When building the graph, you can access the variables like this:
```python
l = Conv2D('conv1', l, 32, 3)
print(l.variables.W)
print(l.variables.b)
```
But note that this is a hacky way and may not work with future versions of TensorFlow.
Also this method doesn't work with LinearWrap, and cannot access the variables created by an activation function.

The output of a layer is usually named `name/output` unless documented differently in the API.
You can always print a tensor to see its name.

### Use Models outside Tensorpack

You can use tensorpack models alone as a simple symbolic function library.
To do this, just enter a [TowerContext](../modules/tfutils.html#tensorpack.tfutils.TowerContext)
when you define your model:
```python
with TowerContext('', is_training=True):
  # call any tensorpack layer
```

Some layers (in particular ``BatchNorm``) has different train/test time behavior which is controlled
by ``TowerContext``. If you need to use the tensorpack version of them in test time, you'll need to create the ops for them under another context.
```python
# Open a `reuse=True` variable scope here if you're sharing variables, then:
with TowerContext('some_name_or_empty_string', is_training=False):
  # build the graph again
```

### Use Other Symbolic Libraries within Tensorpack

When defining the model you can construct the graph using whatever library you feel comfortable with.

Usually, slim/tflearn/tensorlayer are just symbolic function wrappers, calling them is nothing different
from calling `tf.add`. You may need to be careful how regularizations/BN updates are supposed
to be handled in those libraries, though.

It is a bit different to use sonnet/Keras.
sonnet/Keras manages the variable scope by their own model classes, and calling their symbolic functions
always creates new variable scope. See the [Keras example](../examples/keras) for how to use it within tensorpack.
