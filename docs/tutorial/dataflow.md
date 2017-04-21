
# DataFlow

DataFlow is a library to help you build Python iterators to load data.

A DataFlow has a `get_data()` generator method,
which yields `datapoints`.
A datapoint must be a **list** of Python objects which I called the `components` of a datapoint.

For example, to train on MNIST dataset, you can build a DataFlow with a `get_data()` method
that yields datapoints of two elements (components):
a numpy array of shape (64, 28, 28), and an array of shape (64,).

### Composition of DataFlow
One good thing about having a standard interface is to be able to provide
the greatest code reusability.
There are a lot of existing modules in tensorpack which you can use to compose
complex DataFlow instances with a long pre-processing pipeline. A whole pipeline usually
would __read from disk (or other sources), apply augmentations, group into batches,
prefetch data__, etc. A simple example is as the following:

````python
# a DataFlow you implement to produce [image,label] pairs from whatever sources:
df = MyDataFlow(shuffle=True)
# resize the image component of each datapoint
df = AugmentImageComponent(df, [imgaug.Resize((225, 225))])
# group data into batches of size 128
df = BatchData(df, 128)
# start 3 processes to run the dataflow in parallel, and communicate with ZeroMQ
df = PrefetchDataZMQ(df, 3)
````
A more complicated example is the [ResNet training script](../examples/ResNet/imagenet-resnet.py)
with all the data preprocessing.

All these modules are written in Python,
so you can easily implement whatever operations/transformations you need,
without worrying about adding operators to TensorFlow.
In the meantime, thanks to the prefetching, it can still run fast enough for
tasks as large as ImageNet training.

Unless you are working with standard data types (image folders, LMDB, etc),
you would usually want to write your own DataFlow.
See [another tutorial](http://tensorpack.readthedocs.io/en/latest/tutorial/extend/dataflow.html)
for details.

<!--
   - TODO mention RL, distributed data, and zmq operator in the future.
	 -->

### Use DataFlow outside Tensorpack
Another good thing about DataFlow is that it is independent of
tensorpack internals. You can just use it as an efficient data processing pipeline
and plug it into other frameworks.

To use a DataFlow independently, you will need to call `reset_state()` first to initialize it,
and then use the generator however you want:
```python
df = SomeDataFlow()

df.reset_state()
generator = df.get_data()
for dp in generator:
	# dp is now a list. do whatever
```
