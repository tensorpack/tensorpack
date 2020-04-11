
# DataFlow

DataFlow is a pure-Python library to create iterators for efficient data loading.
It is originally part of tensorpack, and now also available as a [separate library](https://github.com/tensorpack/dataflow).

### What is DataFlow

**Definition**: A DataFlow instance is a idiomatic Python iterator object that has a `__iter__()` method
which yields `datapoints`, and optionally a `__len__()` method returning the size of the DataFlow.
A datapoint is a **list or dict** of Python objects, each of which are called the `components` of a datapoint.

**Example**: to train on MNIST dataset, you may need a DataFlow with a `__iter__()` method
that yields datapoints (lists) of two components:
a numpy array of shape (64, 28, 28), and an array of shape (64,).

DataFlow is independent of the training frameworks since it produces any python objects
(usually numpy arrays).
You can simply use DataFlow as a data processing pipeline and plug it into your own training code.

### Load Raw Data
We do not make any assumptions about your data format.
You would usually want to write the source DataFlow (`MyDataFlow` in the example below) for your own data format.
See [another tutorial](extend/dataflow.md) for simple instructions on writing a DataFlow.

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

A list of built-in DataFlow to use can be found at [API docs](../modules/dataflow).
You can also find complicated real-life DataFlow pipelines in the [ImageNet training script](../../examples/ImageNetModels/imagenet_utils.py)
or other tensorpack examples.

### Parallelize the Pipeline

DataFlow includes **carefully optimized** parallel runners and parallel mappers: `Multi{Thread,Process}{Runner,MapData}`.
Runners execute multiple clones of a dataflow in parallel.
Mappers execute a mapping function in parallel on top of an existing dataflow.
You can find details in the [API docs](../modules/dataflow) under the
"parallel" and "parallel_map" section.

[Parallel DataFlow tutorial](parallel-dataflow.md) gives a deeper dive
on how to use them to optimize your data pipeline.

### Run the DataFlow

When training with tensorpack, typically it is the `InputSource` interface that runs the DataFlow.

When using DataFlow alone without tensorpack,
you need to call `reset_state()` first to initialize it,
and then use the generator however you like:

```python
df = SomeDataFlow()

df.reset_state()
for dp in df:
    # dp is now a list/dict. do whatever with it
```

### Why DataFlow?

It's **easy and fast**.
For more discussions, see [Why DataFlow?](./philosophy/dataflow.md)
Nevertheless, using DataFlow is not required in tensorpack.
Tensorpack supports data loading with native TF operators / TF datasets as well.
