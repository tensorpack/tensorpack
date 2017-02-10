
# Dataflow

Dataflow is a unified interface to produce data.

A Dataflow has a `get_data()` generator method,
which yields a `datapoint` when called.
A datapoint must be a **list** of Python objects which I called the `components` of this datapoint.

For example, to train on MNIST dataset, you can build a Dataflow
that produces datapoints of two elements (components):
a numpy array of shape (64, 28, 28), and an array of shape (64,).

### Composition of DataFlow
One good thing about having a standard interface is to be able to provide
the greatest code reusablility.
There are a lot of existing modules in tensorpack which you can use to compose
complex Dataflow instances with a long pre-processing pipeline. A whole pipeline usually
would __read from disk (or other sources), apply augmentations, group into batches,
prefetch data__, etc. An example is as the following:

````python
# define a Dataflow which produces image-label pairs from a caffe lmdb database
df = CaffeLMDB('/path/to/caffe/lmdb', shuffle=False)
# resize the image component of each datapoint
df = AugmentImageComponent(df, [imgaug.Resize((225, 225))])
# group data into batches of size 128
df = BatchData(df, 128)
# start 3 processes to run the dataflow in parallel, and transfer data with ZeroMQ
df = PrefetchDataZMQ(df, 3)
````
A more complicated example is the [ResNet training script](../examples/ResNet/imagenet-resnet.py)
with all the data preprocessing.

All these modules are written in Python,
so you can easily implement whatever opeartions/transformations you need,
without worrying about adding operators to TensorFlow.
In the mean time, thanks to the prefetching, it can still run fast enough for
tasks as large as ImageNet training.

<!--
   - TODO mention RL, distributed data, and zmq operator in the future.
	 -->

### Reuse in other frameworks
Another good thing about Dataflow is that it is independent of
tensorpack internals. You can just use it as an efficient data processing pipeline,
and plug it into other frameworks.

### Write your own Dataflow

There are several existing Dataflow, e.g. ImageFromFile, DataFromList, which you can
use to read images or load data from a list.
But in general, you'll probably need to write a new Dataflow to produce data for your task.
Dataflow implementations for several well-known datasets are provided in the
[dataflow.dataset](http://tensorpack.readthedocs.io/en/latest/modules/tensorpack.dataflow.dataset.html)
module, which you can take as a reference.

A Dataflow has a `get_data()` method which yields a datapoint every time.
```python
class MyDataFlow(DataFlow):
  def get_data(self):
    for k in range(100):
      digit = np.random.rand(28, 28)
      label = np.random.randint(10)
      yield [digit, label]
```

Optionally, Dataflow can implement the following two methods:

+ `size()`. Return the number of elements the generator can produce. Certain modules might require this.
	For example, only Dataflows with the same number of elements can be joined together.

+ `reset_state()`. It's guranteed that the process which uses this DataFlow will invoke this method before using it.
	So if this DataFlow needs to something after a `fork()`, you should put it here.

	A typical situation is when your Dataflow uses random number generator (RNG). Then you'd need to reset the RNG here,
	otherwise child processes will have the same random seed. The `RNGDataFlow` class does this already.

With a "low-level" Dataflow defined, you can then compose it with existing modules.
