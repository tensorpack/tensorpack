
# DataFlow

### What is DataFlow

DataFlow is a pure-Python library to create iterators for efficient data loading.

**Definition**: A DataFlow is a idiomatic Python iterator object that has a `__iter__()` method
which yields `datapoints`, and optionally a `__len__()` method returning the size of the DataFlow.
A datapoint is a **list or dict** of Python objects, each of which are called the `components` of a datapoint.

**Example**: to train on MNIST dataset, you may need a DataFlow with a `__iter__()` method
that yields datapoints (lists) of two components:
a numpy array of shape (64, 28, 28), and an array of shape (64,).

As you saw,
DataFlow is __independent of TensorFlow__ since it produces any python objects
(usually numpy arrays).
To `import tensorpack.dataflow`, you don't even have to install TensorFlow.
You can simply use DataFlow as a data processing pipeline and plug it into any other frameworks.

### Load Raw Data
We do not make any assumptions about your data format.
You would usually want to write the source DataFlow (`MyDataFlow` in the example below) for your own data format.
See [another tutorial](extend/dataflow.html) for simple instructions on writing a DataFlow.

### Assemble the Pipeline
There are a lot of existing DataFlow utilities in tensorpack, which you can use to assemble
the source DataFlow with complex data pipeline.
A common pipeline usually would 
__read from disk (or other sources), 
apply transformations, 
group into batches, prefetch data__, etc, and all __run in parallel__.
A simple DataFlow pipeline is like the following:

````python
# a DataFlow you implement to produce [tensor1, tensor2, ..] lists from whatever sources:
df = MyDataFlow(dir='/my/data', shuffle=True)
# apply transformation to your data
df = MapDataComponent(df, lambda t: transform(t), 0)
# group data into batches of size 128
df = BatchData(df, 128)
# start 3 processes to run the dataflow in parallel
df = MultiProcessRunnerZMQ(df, 3)
````

A list of built-in DataFlow to compose with can be found at [API docs](../modules/dataflow.html).
You can also find complicated real-life DataFlow pipelines in the [ImageNet training script](../examples/ImageNetModels/imagenet_utils.py)
or other tensorpack examples.

### Parallelize the Pipeline

DataFlow includes optimized parallel runner and parallel mapper.
You can find them in the [API docs](../modules/dataflow.html) under the
"parallel" and "parallel_map" section.

The [Efficient DataFlow](efficient-dataflow.html) give a deeper dive
on how to use them to optimize your data pipeline.

### Run the DataFlow

When training with tensorpack, typically it is the `InputSource` interface that runs the DataFlow.
However, DataFlow can be used without other tensorpack components.
To run a DataFlow by yourself, call `reset_state()` first to initialize it,
and then use the generator however you like:

```python
df = SomeDataFlow()

df.reset_state()
for dp in df:
    # dp is now a list. do whatever
```

### Why DataFlow

1. It's easy: write everything in pure Python, and reuse existing utilities.
	 On the contrary, writing data loaders in TF operators is usually painful, and performance is hard to tune.
	 See more discussions in [Python Reader or TF Reader](extend/input-source.html#python-reader-or-tf-reader).
2. It's fast: see [Efficient DataFlow](efficient-dataflow.html)
	on how to build a fast DataFlow with parallelism.
	If you're using DataFlow with tensorpack, also see [Input Pipeline tutorial](extend/input-source.html)
	on how tensorpack further accelerates data loading in the graph.

Nevertheless, tensorpack supports data loading with native TF operators / TF datasets as well.

Read the [API documentation](../../modules/dataflow.html)
to see API details of DataFlow and a complete list of built-in DataFlow.
