
# Why DataFlow?

There are many other data loading solutions for deep learning.
Here we explain why you may want to use Tensorpack DataFlow for your own good:
**it's easy, and fast (enough)**.

### How Fast Do You Actually Need?

Your data pipeline **only needs to be fast enough**.

In practice, you should always first make sure your data pipeline runs
asynchronously with your training.
The method to do so is different in each training framework,
and in tensorpack this is automatically done by the [InputSource](../extend/input-source.md)
interface.

Once you make sure the data pipeline runs async with your training,
the data pipeline only needs to be as fast as the training.
**Getting faster brings no gains** to overall throughput.
It only needs to be fast enough.

If you have used other data loading libraries, you may doubt
how easy it is to make data pipeline fast enough with pure Python.
In fact, it is usually not hard with DataFlow, because it's carefully optimized.

For example: if you train a ResNet-50 on ImageNet,
DataFlow is fast enough for you unless you use
8 V100s with both FP16 and XLA enabled, which most people don't.
For tasks that are less data-hungry (e.g., object detection, or most NLP tasks),
DataFlow is already an overkill.
See the [Efficient DataFlow](../efficient-dataflow.md) tutorial on how
to build a fast Python loader with DataFlow.

There is no reason to try a more complicated solution,
when you don't know whether a simple solution is fast enough.
And for us, we may optimize DataFlow even more, but we just haven't found the reason to do so.

### Which Data Format?

Certain libraries advocate for a new binary data format (e.g., TFRecords, RecordIO).
Do you need to use them?
We think you usually do not, at least not after you try DataFlow, because these
formats are:

1. **Not Easy**: To use the new binary format,
	 you need to write a script, to process your data from its original format,
	 to this new format. Then you read data from this format to training workers.
	 It's a waste of your effort: the intermediate format does not have to exist.

1. **Not Easy**: Even when you do need to use an intermediate format that's different from your
	 original data format
	(for performance reasons, for example), there are many formats you can choose from.

	 Why use a special binary format when you could use something else?
	 A different format may bring you:

	 * Simpler code for data loading.
	 * Easier visualization.
	 * Interoperability with other libraries.
	 * More functionalities.

	 Different formats have their strength and weakness in the above aspects.
	 Forcing a single binary format on users is certainly not ideal.
	 We should let users make the choice.

1. **Not Necessarily Fast**:
	There are cases when having an intermediate format is useful for performance reasons.
	For example, to apply some one-time expensive preprocessing to your dataset.
	But other formats are probably equally fast.

	Formats like TFRecords and RecordIO are just as fast as your disk, and of course,
	as fast as other libraries.
	Decades of engineering in dataset systems have provided
	many other competitive formats like LMDB, HDF5 that are:
	* Equally fast (if not faster)
	* More generic (not tied to your training framework)
	* Providing more features (e.g. random access)

    The only unique benefit a format like TFRecords or RecordIO may give you,
    is the native integration with the training framework, which may bring a
    small gain to speed.

On the other hand, DataFlow is:

1. **Easy**: Any Python function that produces data can be made a DataFlow and
   used for training. No need for intermediate format when you don't.
1. **Flexible**: Since it is in pure Python, you can use any data format.
   When you need, you can still easily serialize your dataflow to a single-file
   format with
   [a few lines of code](../../modules/dataflow.html#tensorpack.dataflow.LMDBSerializer).


### Alternative Data Loading Solutions:

Some frameworks have also provided good framework-specific solutions for data loading.
On the contrary, DataFlow is framework-agnostic: you can use it in any Python environment.
In addition to this benefit, there are other reasons you might prefer DataFlow over the alternatives:

#### tf.data and other graph operations

The huge disadvantage of loading data in a computation graph is obvious:
__it's extremely inflexible__.

Why would you ever want to do anything in a computation graph? Here are the possible reasons:

* Automatic differentiation
* Run the computation on different devices
* Serialize the description of your computation
* Automatic performance optimization

They all make sense for training neural networks, but **not much for data loading**.

Unlike running a neural network model, data processing is a complicated and poorly-structured task.
You need to handle different formats, handle corner cases, noisy data, combination of data.
Doing these requires condition operations, loops, data structures, sometimes even exception handling.
These operations are __naturally not the right task for a symbolic graph__,
and it's hard to debug since it's not Python.

Let's take a look at what users are asking for `tf.data`:
* Different ways to [pad data](https://github.com/tensorflow/tensorflow/issues/13969), [shuffle data](https://github.com/tensorflow/tensorflow/issues/14518)
* [Handle none values in data](https://github.com/tensorflow/tensorflow/issues/13865)
* [Handle dataset that's not a multiple of batch size](https://github.com/tensorflow/tensorflow/issues/13745)
* [Different levels of determinism](https://github.com/tensorflow/tensorflow/issues/13932)
* [Sort/skip some data](https://github.com/tensorflow/tensorflow/issues/14250)
* [Write data to files](https://github.com/tensorflow/tensorflow/issues/15014)

To support all these features which could've been done with __3 lines of code in Python__, you need either a new TF
API, or ask [Dataset.from_generator](https://www.tensorflow.org/versions/r1.4/api_docs/python/tf/contrib/data/Dataset#from_generator)
(i.e. Python again) to the rescue.

It only makes sense to use TF to read data, if your data is originally very clean and well-formatted.
If not, you may feel like writing a Python script to reformat your data, but then you're
almost writing a DataFlow (a DataFlow can be made from a Python iterator)!

As for speed, when TF happens to support and optimize the operators you need,
it does offer a similar or higher speed (it takes effort to tune, of course).
But how do you make sure you'll not run into one of the unsupported situations listed above?

#### torch.utils.data.{Dataset,DataLoader}

In the design, `torch.utils.data.Dataset` is simply a Python container/iterator, similar to DataFlow.
However it has made some **bad assumptions**:
it assumes your dataset supports `__getitem__`,
which does not work when you have a dynamic/unreliable data source,
or when you need to filter your data on the fly.

`torch.utils.data.DataLoader` is quite good, despite that it also makes some
**bad assumptions on batching** and is not always efficient:

1. `torch.utils.data.DataLoader` assumes that:
   1. You do batch training
   1. You use a constant batch size
   1. Indices are sufficient to determine which samples to batch together
   1. Multiprocessing is a better parallelization strategy than multithreading.

   None of these are necessarily true.

2. Its multiprocessing implementation is efficient on `torch.Tensor`,
   but inefficient for generic data type or numpy arrays.
   Also, its implementation [does not always clean up the subprocesses correctly](https://github.com/pytorch/pytorch/issues/16608).

PyTorch starts to improve on these bad assumptions (e.g., with [IterableDataset](https://github.com/pytorch/pytorch/pull/19228)).
On the other hand, DataFlow:

1. Is an iterator, not necessarily has a length or can be indexed. This is more generic.
2. Does not assume batches, and allow you to implement different batching logic easily.
3. Is optimized for generic data type and numpy arrays.


```eval_rst
.. note:: An iterator interface is more generic than ``__getitem__``?

   DataFlow's iterator interface can perfectly simulate the behavior of indexing interface like this:

    .. code-block:: python

        # A dataflow which produces indices, like [0], [1], [2], ...
        # The indices can be either sequential, or more fancy, akin to torch.utils.data.Sampler.
        df = SomeIndexGenerator()
        # Map the indices to datapoints by ``__getitem__``.
        df = MapData(df, lambda idx: dataset[idx[0]])
```
