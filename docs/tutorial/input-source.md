
# Input Pipeline

This tutorial contains some general discussions on the topic of
"how to read data efficiently to work with TensorFlow",
and how tensorpack supports these methods.
You don't have to read it because these are details under the tensorpack interface,
but knowing it could help understand the efficiency and choose the best input pipeline for your task.

## Prepare Data in Parallel

![prefetch](https://cloud.githubusercontent.com/assets/1381301/26525192/36e5de48-4304-11e7-88ab-3b790bd0e028.png)

A common sense no matter what framework you use:
<center>
Prepare data in parallel with the training!
</center>

The reasons are:
1. Data preparation often consumes non-trivial time (depend on the actual problem).
2. Data preparation often uses completely different resources from training (see figure above) --
	doing them together doesn't slow you down. In fact you can further parallelize different stages in
	the preparation since they also use different resources.
3. Data preparation often doesn't depend on the result of the previous training step.

Let's do some simple math: according to [tensorflow/benchmarks](https://www.tensorflow.org/performance/benchmarks),
4 P100 GPUs can train ResNet50 at 852 images/sec, and the size of those images are 852\*224\*224\*3\*4bytes = 489MB.
Assuming you have 5GB/s `memcpy` bandwidth (roughly like this if you run single-thread copy), simply copying the data once would take 0.1s -- slowing
down your training by 10%. Think about how many more copies are made during your preprocessing.

Failure to hide the data preparation latency is the major reason why people
cannot see good GPU utilization. __Always choose a framework that allows latency hiding.__
However most other TensorFlow wrappers are designed to be `feed_dict` based.
This is the major reason why tensorpack is [faster](https://github.com/tensorpack/benchmarks).

## Python Reader or TF Reader ?

The above discussion is valid regardless of what you use to load/preprocess data,
either Python code or TensorFlow operators.
Both are supported in tensorpack, while we recommend using Python.

### TensorFlow Reader: Pros
* Faster read/preprocessing.

	* Potentially true, but not necessarily. With Python you can call a variety of other fast libraries, which
		you might not have a good support in TF. For example, LMDB could be faster than TFRecords.
	* Python may be just fast enough.

		As long as data preparation runs faster than training, and the latency of all four blocks in the
		above figure is hidden, it makes no difference at all.
		For most types of problems, up to the scale of multi-GPU ImageNet training,
		Python can offer enough speed if you use a fast library (e.g. `tensorpack.dataflow`).
		See the [Efficient DataFlow](efficient-dataflow.html) tutorial on how to build a fast Python reader with DataFlow.

* No "Copy to TF" (i.e. `feed_dict`) stage.

	* True. But as mentioned above, the latency can usually be hidden.

		In tensorpack, TF queues are used to hide the "Copy to TF" latency,
		and TF `StagingArea` can help hide the "Copy to GPU" latency.
		They are used by most examples in tensorpack.

### TensorFlow Reader: Cons
The disadvantage of TF reader is obvious and it's huge: it's __too complicated__.

Unlike running a mathematical model, reading data is a complicated and badly-structured task.
You need to handle different data format, handle corner cases in noisy data,
which all require condition operations, loops, sometimes even exception handling. These operations
are __naturally not suitable__ for a symbolic graph.

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

It only makes sense to use TF to read data, if your data is originally very clean and well-formated.
If not, you may feel like writing a script to format your data, but then you're almost writing a Python loader already!

Think about it: it's a waste of time to write a Python script to transform from raw data to TF-friendly format,
then a TF script to transform from this format to tensors.
The intermediate format doesn't have to exist.
You just need the right interface to connect Python to the graph directly, efficiently.
`tensorpack.InputSource` is such an interface.

## InputSource

`InputSource` is an abstract interface in tensorpack, to describe where the inputs come from and how they enter the graph.
For example,

1. [FeedInput](../modules/input_source.html#tensorpack.input_source.FeedInput):
	Come from a DataFlow and get fed to the graph (slow).
2. [QueueInput](../modules/input_source.html#tensorpack.input_source.QueueInput):
  Come from a DataFlow and get prefetched on CPU by a TF queue.
3. [StagingInput](../modules/input_source.html#tensorpack.input_source.StagingInput):
	Come from some `InputSource`, then prefetched on GPU by a TF StagingArea.
4. [TFDatasetInput](http://tensorpack.readthedocs.io/en/latest/modules/input_source.html#tensorpack.input_source.TFDatasetInput)
	Come from a `tf.data.Dataset`.
5. [dataflow_to_dataset](http://tensorpack.readthedocs.io/en/latest/modules/input_source.html#tensorpack.input_source.TFDatasetInput.dataflow_to_dataset)
	Come from a DataFlow, and further processed by `tf.data.Dataset`.
6. [TensorInput](../modules/input_source.html#tensorpack.input_source.TensorInput):
	Come from some tensors you wrote.
7. [ZMQInput](http://tensorpack.readthedocs.io/en/latest/modules/input_source.html#tensorpack.input_source.ZMQInput)
	Come from some ZeroMQ pipe, where the load/preprocessing may happen on a different machine.
