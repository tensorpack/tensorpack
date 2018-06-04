
## Write a Layer

The first thing to note: __you never have to write a layer__.
Tensorpack layers are nothing but wrappers of symbolic functions.
In tensorpack, you can use __any__ symbolic functions you have written or seen elsewhere with or without tensorpack layers.

If you would like, you can make a symbolic function become a "layer" by following some simple rules, and then gain benefits from the framework.

Take a look at the [Convolutional Layer](../../tensorpack/models/conv2d.py#L14) implementation for an example of how to define a layer:

```python
@layer_register(log_shape=True)
def Conv2D(x, out_channel, kernel_shape,
           padding='SAME', stride=1,
           W_init=None, b_init=None,
           nl=tf.nn.relu, split=1, use_bias=True):
```

Basically, a tensorpack layer is just a symbolic function, but with the following rules:

+ It is decorated by `@layer_register`.
+ The first argument is its "input". It must be a **tensor or a list of tensors**.
+ It returns either a tensor or a list of tensors as its "output".


By making a symbolic function a "layer", the following things will happen:
+ You will need to call the function with a scope name as the first argument, e.g. `Conv2D('conv0', x, 32, 3)`.
	Everything happening in this function will be under the variable scope `conv0`.
	You can register the layer with `use_scope=False` to disable this feature.
+ Static shapes of input/output will be printed to screen (if you register with `log_shape=True`).
+ `argscope` will work for all its arguments except the input tensor(s).
+ It will work with `LinearWrap`: you can use it if the output of one layer matches the input of the next layer.

There is no rule about what kind of symbolic functions should be made a layer -- they are quite
similar anyway. However, in general, I define the following symbolic functions as layers:
+ Functions which contain variables. A variable scope is almost always helpful for such functions.
+ Functions which are commonly referred to as "layers", such as pooling. This makes a model
	definition more straightforward.

