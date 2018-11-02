
# DataFlow

### What is DataFlow

DataFlow is a library to build Python iterators for efficient data loading.

**Definition**: A DataFlow is a idiomatic Python container object that has a `__iter__()` generator method, 
which yields `datapoints` and optionally a `__len__()` method returning the size of the flow.
A datapoint is a **list** of Python objects which are called the `components` of a datapoint.

**Example**: to train on MNIST dataset, you may need a DataFlow with a `__iter__()` method
that yields datapoints (lists) of two components:
a numpy array of shape (64, 28, 28), and an array of shape (64,).

As you saw,
DataFlow is __independent of TensorFlow__ since it produces any python objects
(usually numpy arrays).
To `import tensorpack.dataflow`, you don't even have to install TensorFlow.
You can simply use DataFlow as a data processing pipeline and plug it into any other frameworks.


### Composition of DataFlow
One good thing about having a standard interface is to be able to provide
the greatest code reusability.
There are a lot of existing DataFlow utilities in tensorpack, which you can use to compose
DataFlow with complex data pipeline. A common pipeline usually
would __read from disk (or other sources), apply transformations, group into batches,
prefetch data__, etc. A simple example is as the following:

````python
# a DataFlow you implement to produce [tensor1, tensor2, ..] lists from whatever sources:
df = MyDataFlow(dir='/my/data', shuffle=True)
# resize the image component of each datapoint
df = AugmentImageComponent(df, [imgaug.Resize((225, 225))])
# group data into batches of size 128
df = BatchData(df, 128)
# start 3 processes to run the dataflow in parallel
df = PrefetchDataZMQ(df, 3)
````
You can find more complicated DataFlow in the [ImageNet training script](../examples/ImageNetModels/imagenet_utils.py)
with all the data preprocessing.

### Work with Your Data
Unless you are working with standard data types (image folders, LMDB, etc),
you would usually want to write the source DataFlow (`MyDataFlow` in the above example) for your data format.
See [another tutorial](extend/dataflow.html) for simple instructions on writing a DataFlow.
Once you have the source reader, all the [existing DataFlows](../modules/dataflow.html) are ready for you to complete
the rest of the data pipeline.

### Why DataFlow

1. It's easy: write everything in pure Python, and reuse existing utilities.
	 On the contrary, writing data loaders in TF operators is usually painful, and performance is hard to tune.
	 See more discussions in [Python Reader or TF Reader](input-source.html#python-reader-or-tf-reader).
2. It's fast: see [Efficient DataFlow](efficient-dataflow.html)
	on how to build a fast DataFlow with parallelism.
	If you're using DataFlow with tensorpack, also see [Input Pipeline tutorial](input-source.html)
	on how tensorpack further accelerates data loading in the graph.

Nevertheless, tensorpack supports data loading with native TF operators / TF datasets as well.

### Use DataFlow outside Tensorpack

Normally, tensorpack `InputSource` interface links DataFlow to the graph for training.
If you use DataFlow in other places such as your custom code, call `reset_state()` first to initialize it,
and then use the generator however you like:
```python
df = SomeDataFlow()

df.reset_state()
for dp in df:
	# dp is now a list. do whatever
```

Read the [API documentation](../../modules/dataflow.html#tensorpack.dataflow.DataFlw)
to see API details of DataFlow.
