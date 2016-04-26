
## Dataflow

Dataflow uses Python generator to produce data.

A Dataflow has to implement the `get_data()` generator method, which generates a `datapoint` when called.
A datapoint must be a list of picklable Python object, each is called a `component` of the datapoint.
For example to train on MNIST dataset, you can define a Dataflow that produces datapoints of shape `[(BATCH, 28, 28), (BATCH,)]`.

Then, multiple Dataflows can be composed together to build a complex data-preprocessing pipeline,
including __reading from disk, batching, augmentations, prefetching__, etc. These components written in Python
can provide a more flexible data pipeline than with TensorFlow operators.
Take a look at [common Dataflow](../../tensorpack/dataflow/common.py) and a [example of use](../../examples/ResNet/cifar10-resnet.py#L125).

Optionally, Dataflow can implement the following two methods:

+ `size()`. Return the number of elements. Some components in the pipeline might require this to be
	implemented. For example, only Dataflows with the same number of elements can be [joined](../../tensorpack/dataflow/common.py#L276).

+ `reset_state()`. It's necessary if your Dataflow uses RNG. This
	method should reset the state of RNG and will be called after a fork, so that different child
	processes won't produce identical data.


NOTE: Dataflow aims to be independent of tensorflow.
It should be useful for other python-based learning libraries as well.

Common public datasets are also a kind of Dataflow. Some are defined in [dataflow.dataset](../../tensorpack/dataflow/dataset).
