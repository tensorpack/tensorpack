
# Input Pipeline

This tutorial contains some general discussions on the topic of
"how to read data efficiently to work with TensorFlow",
and how tensorpack supports these methods.
As a beginner you can skip this tutorial, because these are details under the tensorpack interface,
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
cannot see good GPU utilization. You should __always choose a framework that enables latency hiding.__
However most other TensorFlow wrappers are designed to be `feed_dict` based.
Tensorpack has built-in mechanisms to hide latency of the above stages.
This is one of the reasons why tensorpack is [faster](https://github.com/tensorpack/benchmarks).

## Python Reader or TF Reader ?

The above discussion is valid regardless of what you use to load/preprocess data,
either Python code or TensorFlow operators, or a mix of two.
Both are supported in tensorpack, while we recommend using Python.

### TensorFlow Reader: Pros

People often think they should use `tf.data` because it's fast.

* Indeed it's often fast, but not necessarily. With Python you have access to many other fast libraries, which might be unsupported in TF.
* Python may be just fast enough.

    As long as data preparation keeps up with training, and the latency of all four blocks in the
    above figure is hidden, __faster reader brings no gains to overall throughput__.
    For most types of problems, up to the scale of multi-GPU ImageNet training,
    Python can offer enough speed if you use a fast library (e.g. `tensorpack.dataflow`).
    See the [Efficient DataFlow](efficient-dataflow.html) tutorial on how to build a fast Python reader with DataFlow.

### TensorFlow Reader: Cons
The disadvantage of TF reader is obvious and it's huge: it's __too complicated__.

Unlike running a mathematical model, data processing is a complicated and poorly-structured task.
You need to handle different formats, handle corner cases, noisy data, combination of data.
Doing these requires condition operations, loops, data structures, sometimes even exception handling.
These operations are __naturally not the right task for a symbolic graph__.

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

Think about it: it's a waste of time to write a Python script to transform from some format to TF-friendly format,
then a TF script to transform from this format to tensors.
The intermediate format doesn't have to exist.
You just need the right interface to connect Python to the graph directly, efficiently.
`tensorpack.InputSource` is such an interface.

## InputSource

`InputSource` is an abstract interface used by tensorpack trainers, to describe where the inputs come from and how they enter the graph.
Some choices are:

1. [FeedInput](../modules/input_source.html#tensorpack.input_source.FeedInput):
	Data come from a DataFlow and get fed to the graph (slow).
2. [QueueInput](../modules/input_source.html#tensorpack.input_source.QueueInput):
    Data come from a DataFlow and get buffered on CPU by a TF queue.
3. [StagingInput](../modules/input_source.html#tensorpack.input_source.StagingInput):
	Come from some other `InputSource`, then prefetched on GPU by a TF StagingArea.
4. [TFDatasetInput](../modules/input_source.html#tensorpack.input_source.TFDatasetInput)
	Come from a `tf.data.Dataset`.
5. [dataflow_to_dataset](../modules/input_source.html#tensorpack.input_source.TFDatasetInput.dataflow_to_dataset)
	Come from a DataFlow, and then lfurther processed by utilities in `tf.data.Dataset`.
6. [TensorInput](../modules/input_source.html#tensorpack.input_source.TensorInput):
	Come from some tensors you define (can be reading ops, for example).
7. [ZMQInput](../modules/input_source.html#tensorpack.input_source.ZMQInput)
	Come from some ZeroMQ pipe, where the reading/preprocessing may happen in a different process or even a different machine.

Typically, we recommend using `DataFlow + QueueInput` as it's good for most use cases.
If your data has to come from a separate process for whatever reasons, use `ZMQInput`.
If you need to use TF reading ops directly, either define a `tf.data.Dataset`
and use `TFDatasetInput`, or use `TensorInput`.

Refer to the documentation of these `InputSource` for more details.
