# Efficient DataFlow

This tutorial gives an overview of how to build an efficient DataFlow, using ImageNet dataset as an example.
Our goal in the end is to have
a __Python generator__ which yields preprocessed ImageNet images and labels as fast as possible.
Since it is simply a generator interface, you can use the DataFlow in any Python-based frameworks (e.g. PyTorch, Keras)
or your own code as well.


**What we are going to do**: We'll use ILSVRC12 dataset, which contains 1.28 million images.
The original images (JPEG compressed) are 140G in total.
The average resolution is about 400x350 <sup>[[1]]</sup>.
Following the [ResNet example](../../examples/ResNet), we need images in their original resolution,
so we will read the original dataset (instead of a down-sampled version), and
then apply complicated preprocessing to it.
We hope to reach a speed of **1k~5k images per second**, to keep GPUs busy.

Some things to know before reading:

1. You are recommended to read the [Parallel DataFlow Tutorial](./parallel-dataflow.md) first.
1. You only need the data loader to be **fast enough, but not faster**.
   See [How Fast Do You Actually Need](philosophy/dataflow.html#how-fast-do-you-actually-need) for details.
   For smaller datasets (e.g. several GBs of images with lightweight preprocessing),
   a simple reader plus some multiprocess runner is usually fast enough.
1. Having a fast Python generator **alone** may or may not improve your overall training speed.
   You need mechanisms to hide the latency of **all** preprocessing stages, as mentioned in the
   [InputSource tutorial](./extend/input-source.md).
1. Reading training set and validation set are different.
   In training it's OK to reorder, regroup, or even duplicate some datapoints, as long as the
   data distribution stays the same.
   But in validation we often need the exact set of data, to be able to compute a correct and comparable score.
   This will affect how we build the DataFlow.
1. The actual performance would depend on not only the disk, but also memory (for caching) and CPU (for data processing).
  You may need to tune the parameters (#processes, #threads, size of buffer, etc.)
  or change the pipeline for new tasks and new machines to achieve the best performance.
    The solutions in this tutorial may not help you.
    To improve your own DataFlow, read the
    [performance tuning tutorial](performance-tuning.html#investigate-dataflow)
    before performing or asking about any actual optimizations.

The benchmark code for this tutorial can be found in [tensorpack/benchmarks](https://github.com/tensorpack/benchmarks/tree/master/ImageNet),
including comparison with a similar pipeline built with `tf.data`.

This tutorial could be a bit complicated for people new to system architectures,
but you do need these to be able to run fast enough on ImageNet-scale dataset.

## Random Read

### Basic
We start from a simple DataFlow:
```python
from tensorpack.dataflow import *
ds0 = dataset.ILSVRC12('/path/to/ILSVRC12', 'train', shuffle=True)
ds1 = BatchData(ds0, 256, use_list=True)
TestDataSpeed(ds1).start()
```

Here `ds0` reads original images from the filesystem. It is implemented simply by:
```python
for filename, label in np.random.shuffle(filelist):
  yield [cv2.imread(filename), label]
```

And `ds1` batch the datapoints from `ds0`, so that we can measure the speed of this DataFlow in terms of "batches per second".
By default, `BatchData` should stack the datapoints into an `numpy.ndarray`,
but since original ImageNet images are of different shapes, we use
`use_list=True** so that it produces lists for now.

Here we're mesuring the time to (1) read from file system speed and (2) decode the image.
On a good filesystem you probably can already observe good speed here (e.g. 5 it/s, that is 1280 images/s), but on HDD the speed may be just 1 it/s,
because we are doing heavy random read on the filesystem (regardless of whether `shuffle` is True).
Image decoding in `cv2.imread` may also be a bottleneck at this early stage, since it requires a fast CPU.

### Parallel Runner

We will now add the cheapest pre-processing now to get an ndarray in the end instead of a list
(because training will need ndarray eventually):

```eval_rst

.. code-block:: python
    :emphasize-lines: 2,3

    ds = dataset.ILSVRC12('/path/to/ILSVRC12', 'train', shuffle=True)
    ds = AugmentImageComponent(ds, [imgaug.Resize(224)])
    ds = BatchData(ds, 256)

```

You'll start to observe slow down after adding more pre-processing (such as those in the [ResNet example](../../examples/ImageNetModels/imagenet_utils.py)).
Now it's time to add threads or processes:
```eval_rst
.. code-block:: python
    :emphasize-lines: 3

    ds0 = dataset.ILSVRC12('/path/to/ILSVRC12', 'train', shuffle=True)
    ds1 = AugmentImageComponent(ds0, lots_of_augmentors)
    ds = MultiProcessRunnerZMQ(ds1, num_proc=25)
    ds = BatchData(ds, 256)
```

Here we fork 25 processes to run `ds1`, and collect their output through ZMQ IPC protocol.
You can also apply parallel runner after batching, of course.

### Parallel Map
The above DataFlow might be fast, but since it forks the ImageNet reader (`ds0`),
it's **not a good idea to use it for validation** (for reasons mentioned at top.
More details at the [Parallel DataFlow Tutorial](./parallel-dataflow.md) and the [documentation](../modules/dataflow.html#tensorpack.dataflow.MultiProcessRunnerZMQ)).
Alternatively, you can use parallel mapper like this:

```eval_rst
.. code-block:: python
    :emphasize-lines: 3-6

    ds0 = dataset.ILSVRC12('/path/to/ILSVRC12', 'train', shuffle=True)
    augmentor = AugmentorList(lots_of_augmentors)
    ds1 = MultiThreadMapData(
        ds0, num_thread=25,
        map_func=lambda dp: [augmentor.augment(dp[0]), dp[1]], buffer_size=1000)
    # ds1 = MultiProcessRunnerZMQ(ds1, num_proc=1)
    ds = BatchData(ds1, 256)
```
`MultiThreadMapData` launches a thread pool to fetch data and apply the mapping function on **a single
instance of** `ds0`. This is done by an intermediate buffer of size 1000 to hide the mapping latency.
To reduce the effect of GIL to your main training thread, you want to uncomment the line so that everything above it (including all the
threads) happen in an independent process.

There is no answer whether it is faster to use threads or processes.
Processes avoid the cost of GIL but bring the cost of communication.
You can also try a combination of both (several processes each with several threads),
but be careful of how forks affect your data distribution.

The above DataFlow still has a potential performance problem: only one thread is doing `cv2.imread`.
If you identify this as a bottleneck, you can also use:

```eval_rst
.. code-block:: python
    :emphasize-lines: 5-6

    ds0 = dataset.ILSVRC12Files('/path/to/ILSVRC12', 'train', shuffle=True)
    augmentor = AugmentorList(lots_of_augmentors)
    ds1 = MultiThreadMapData(
        ds0, num_thread=25,
        map_func=lambda dp:
          [augmentor.augment(cv2.imread(dp[0], cv2.IMREAD_COLOR)), dp[1]],
        buffer_size=1000)
    ds1 = MultiProcessRunnerZMQ(ds1, num_proc=1)
    ds = BatchData(ds1, 256)
```

Let's summarize what the above dataflow does:
1. One thread iterates over a shuffled list of (filename, label) pairs, and put them into a queue of size 1000.
2. 25 worker threads take pairs and make them into (preprocessed image, label) pairs.
3. Both 1 and 2 happen together in a separate process, and the results are sent back to main process through ZeroMQ.
4. Main process makes batches, and other tensorpack modules will then take care of how they should go into the graph.

And, of course, there is also a `MultiProcessMapData` as well for you to use.

## Sequential Read

### Save and Load a Single-File DataFlow
Random read may not be a good idea when the data is not on an SSD.
In this case, we can also dump the dataset into one single LMDB file and read it sequentially.

```python
from tensorpack.dataflow import *
class BinaryILSVRC12(dataset.ILSVRC12Files):
    def __iter__(self):
        for fname, label in super(BinaryILSVRC12, self).__iter__():
            with open(fname, 'rb') as f:
                jpeg = f.read()
            jpeg = np.asarray(bytearray(jpeg), dtype='uint8')
            yield [jpeg, label]
ds0 = BinaryILSVRC12('/path/to/ILSVRC/', 'train')
ds1 = MultiProcessRunnerZMQ(ds0, num_proc=1)
LMDBSerializer.save(ds1, '/path/to/ILSVRC-train.lmdb')
```
The above script builds a DataFlow which produces jpeg-encoded ImageNet data.
We store the jpeg string as a numpy array because the function `cv2.imdecode` later expect this format.
Please note we can only use 1 runner process to speed up. If `num_proc>1`, `ds1` will take data
from several forks of `ds0`, then neither the content nor the order of `ds1` will be the same as `ds0`.
See [documentation](../modules/dataflow.html#tensorpack.dataflow.MultiProcessRunnerZMQ)
about caveats of `MultiProcessRunnerZMQ`.

This will generate a database file of 140G. We load the DataFlow back by reading this LMDB file sequentially:
```
ds = LMDBSerializer.load('/path/to/ILSVRC-train.lmdb', shuffle=False)
ds = BatchData(ds, 256, use_list=True)
TestDataSpeed(ds).start()
```
Depending on whether the OS has cached the file for you (and how large the RAM is), the above script
can run at a speed of 10~130 it/s, roughly corresponding to 250MB~3.5GB/s bandwidth. You can test
your cached and uncached disk read bandwidth with `sudo hdparm -Tt /dev/sdX`.
As a reference, on Samsung SSD 850, the uncached speed is about 16it/s.

```eval_rst
.. code-block:: python
    :emphasize-lines: 2

    ds = LMDBSerializer.load('/path/to/ILSVRC-train.lmdb', shuffle=False)
    ds = LocallyShuffleData(ds, 50000)
    ds = BatchData(ds, 256, use_list=True)
```
Instead of shuffling all the training data in every epoch (which would require random read),
the added line above maintains a buffer of datapoints and shuffle them once a while.
It will not affect the model very much as long as the buffer is large enough,
but it can be memory-consuming if buffer is too large.

### Augmentations & Parallelism

Then we add necessary transformations:
```eval_rst
.. code-block:: python
    :emphasize-lines: 3-5

    ds = LMDBSerializer.load(db, shuffle=False)
    ds = LocallyShuffleData(ds, 50000)
    ds = MapDataComponent(ds, lambda x: cv2.imdecode(x, cv2.IMREAD_COLOR), 0)
    ds = AugmentImageComponent(ds, lots_of_augmentors)
    ds = BatchData(ds, 256)
```

1. First we deserialize the datapoints (from raw bytes to [jpeg bytes, label] -- what we dumped in `RawILSVRC12`)
2. Use OpenCV to decode the first component (jpeg bytes) into ndarray
3. Apply augmentations to the ndarray

Both imdecode and the augmentors can be quite slow. We can parallelize them like this:
```eval_rst
.. code-block:: python
    :emphasize-lines: 4,5,6

    ds = LMDBSerializer.load(db, shuffle=False)
    ds = LocallyShuffleData(ds, 50000)

    def f(jpeg_str, label):
        return lots_of_augmentors.augment(cv2.imdecode(x, cv2.IMREAD_COLOR)), label
    ds = MultiProcessMapDataZMQ(ds, num_proc=25, f)
    ds = BatchData(ds, 256)
```

Let me summarize what this DataFlow does:

1. One process reads LMDB file, shuffle them in a buffer and put them into a ZMQ pipe (used by `MultiProcessMapDataZMQ`).
2. 25 processes take items from the queue, decode and process them into [image, label] pairs, and
   send them through ZMQ IPC pipe to the main process.
3. The main process takes data from the pipe, makes batches.

The two DataFlow mentioned in this tutorial (both random read and sequential read) can run at a speed of 1k ~ 5k images per second,
depend on your hardware condition of CPUs, RAM, disks, and the amount of augmentation.
As a reference, tensorpack can train ResNet-18 at 1.2k images/s on 4 old TitanX.
8 V100s can train ResNet-50 at 2.8k images/s according to [tensorpack benchmark](https://github.com/tensorpack/benchmarks/tree/master/ResNet-MultiGPU).
So DataFlow will not be a bottleneck if configured properly.

## Distributed DataFlow

To further scale your DataFlow, you can even run it on multiple machines and collect them on the
training machine. E.g.:
```python
# Data Machine #1, process 1-20:
df = MyLargeData()
send_dataflow_zmq(df, 'tcp://1.2.3.4:8877')
```
```python
# Data Machine #2, process 1-20:
df = MyLargeData()
send_dataflow_zmq(df, 'tcp://1.2.3.4:8877')
```
```python
# Training Machine, process 1-10:
df = MyLargeData()
send_dataflow_zmq(df, 'ipc://@my-socket')
```
```python
# Training Machine, training process
df = RemoteDataZMQ('ipc://@my-socket', 'tcp://0.0.0.0:8877')
TestDataSpeed(df).start()
```


## Common Issues on Windows:

1. Windows does not support IPC protocol of ZMQ. You can only use `MultiProcessRunner`,
   `MultiThreadRunner`, and `MultiThreadMapData`. But you cannot use
   `MultiProcessRunnerZMQ` or `MultiProcessMapData` (which is an alias of `MultiProcessMapDataZMQ`).
2. Windows needs to pickle your dataflow to run it in multiple processes.
   As a result you cannot use lambda functions for mappings, like the examples above.
   You need to create a function in global scope, or a function-like object to perform the mapping.
   This issue also exist on Linux when you do not use the 'fork' start method.

[1]: #ref

<div id=ref> </div>

[[1]]. [ImageNet: A Large-Scale Hierarchical Image Database](http://www.image-net.org/papers/imagenet_cvpr09.pdf), CVPR09
