
## Model Primitives

`models` in tensorpack contains a collection of common model primitives.
Take a look at the [Convolutional Layer](../../tensorpack/models/conv2d.py#L14) implementation for an example of how to define a
model primitive:

```python
@layer_register()
def Conv2D(x, out_channel, kernel_shape,
           padding='SAME', stride=1,
           W_init=None, b_init=None,
           nl=tf.nn.relu, split=1, use_bias=True):
```

A primitive has the same interface as a tensorflow symbolic function: it takes a symbolic input `x` with
some parameters, and return some symbolic outputs.

`@layer_register()` will make this symbolic function become a `layer`, with the following benefits:

+ A variable scope for everything happening in this function.
+ Auto-inferred input/output shapes can be logged to terminal.
+ Work with `argscope` to define default arguments in a simple way.

Some convention when working with a primitive defined under `@layer_register()`:

+ The input must be the first argument in the signature so that logging will know. It can be either a Tensor or a list of Tensor.
+ When called, the first argument should be the name scope and the second be the input.

Take a look at the [Inception example](../../examples/Inception/inception-bn.py#L36) to see how a complicated model can be described with these primitives.

