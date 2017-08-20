# Efficient DataFlow

This tutorial gives an overview of how to build an efficient DataFlow, using ImageNet
dataset as an example.
Our goal in the end is to have
a __Python generator__ which yields preprocessed ImageNet images and labels as fast as possible.
Since it is simply a generator interface, you can use the DataFlow in other Python-based frameworks (e.g. Keras)
or your own code as well.

**What we are going to do**: We'll use ILSVRC12 dataset, which contains 1.28 million images.
The original images (JPEG compressed) are 140G in total.
The average resolution is about 400x350 <sup>[[1]]</sup>.
Following the [ResNet example](../examples/ResNet), we need images in their original resolution,
so we will read the original dataset (instead of a down-sampled version), and
then apply complicated preprocessing to it.
We will need to reach a speed of, roughly **1k ~ 2k images per second**, to keep GPUs busy.

Some things to know before reading:
1. For smaller datasets (e.g. several GBs of images with lightweight preprocessing), a simple reader plus some prefetch should usually work well enough.
	 Therefore you don't have to understand this tutorial in depth unless you really find your data being the bottleneck.
	 This tutorial could be a bit complicated for people new to system architectures, but you do need these to be able to run fast enough on ImageNet-sized dataset.
2. Having a fast Python generator **alone** may or may not improve your overall training speed.
	 You need mechanisms to hide the latency of **all** preprocessing stages, as mentioned in the
	 [previous tutorial](input-source.html).
3. Reading training set and validation set are different.
	 In training it's OK to reorder, regroup, or even duplicate some datapoints, as long as the
	 data distribution roughly stays the same.
	 But in validation we often need the exact set of data, to be able to compute a correct and comparable score.
	 This will affect how we build the DataFlow.
4. The actual performance would depend on not only the disk, but also memory (for caching) and CPU (for data processing).
	 You may need to tune the parameters (#processes, #threads, size of buffer, etc.)
	 or change the pipeline for new tasks and new machines to achieve the best performance.

## Random Read

We start from a simple DataFlow:
```python
from tensorpack.dataflow import *
ds0 = dataset.ILSVRC12('/path/to/ILSVRC12', 'train', shuffle=True)
ds1 = BatchData(ds0, 256, use_list=True)
TestDataSpeed(ds1).start()
```

Here `ds0` reads original images from the filesystem. It is implemented simply by:
```python
for filename, label in filelist:
  yield [cv2.imread(filename), label]
```

And `ds1` batch the datapoints from `ds0`, so that we can measure the speed of this DataFlow in terms of "batch per second".
By default, `BatchData`
will stack the datapoints into an `numpy.ndarray`, but since original images are of different shapes, we use
`use_list=True` so that it just produces lists.

On a good filesystem you probably can already observe good speed here (e.g. 5 it/s, that is 1280 images/s), but on HDD the speed may be just 1 it/s,
because we are doing heavy random read on the filesystem (regardless of whether `shuffle` is True).
Image decoding in `cv2.imread` could also be a bottleneck at this early stage.

We will now add the cheapest pre-processing now to get an ndarray in the end instead of a list
(because training will need ndarray eventually):
```eval_rst
.. code-block:: python
	  :emphasize-lines: 2,3

		ds = dataset.ILSVRC12('/path/to/ILSVRC12', 'train', shuffle=True)
		ds = AugmentImageComponent(ds, [imgaug.Resize(224)])
		ds = BatchData(ds, 256)
```
You'll start to observe slow down after adding more pre-processing (such as those in the [ResNet example](../examples/ResNet/imagenet_resnet_utils.py)).
Now it's time to add threads or processes:
```eval_rst
.. code-block:: python
	  :emphasize-lines: 3

		ds0 = dataset.ILSVRC12('/path/to/ILSVRC12', 'train', shuffle=True)
		ds1 = AugmentImageComponent(ds0, lots_of_augmentors)
		ds = PrefetchDataZMQ(ds1, nr_proc=25)
		ds = BatchData(ds, 256)
```
Here we start 25 processes to run `ds1`, and collect their output through ZMQ IPC protocol,
which is faster than `multiprocessing.Queue`. You can also apply prefetch after batch, of course.

The above DataFlow might be fast, but since it forks the ImageNet reader (`ds0`),
it's **not a good idea to use it for validation** (for reasons mentioned at top).
Alternatively, you can use multi-threaded preprocessing like this:

```eval_rst
.. code-block:: python
	  :emphasize-lines: 3-6

		ds0 = dataset.ILSVRC12('/path/to/ILSVRC12', 'train', shuffle=True)
		augmentor = AugmentorList(lots_of_augmentors)
		ds1 = ThreadedMapData(
				ds0, nr_thread=25,
				map_func=lambda dp: [augmentor.augment(dp[0]), dp[1]], buffer_size=1000)
		# ds1 = PrefetchDataZMQ(ds1, nr_proc=1)
		ds = BatchData(ds1, 256)
```
`ThreadedMapData` launches a thread pool to fetch data and apply the mapping function on **a single
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
		ds1 = ThreadedMapData(
				ds0, nr_thread=25,
				map_func=lambda dp:
					[augmentor.augment(cv2.imread(dp[0], cv2.IMREAD_COLOR)), dp[1]],
				buffer_size=1000)
		ds1 = PrefetchDataZMQ(ds1, nr_proc=1)
		ds = BatchData(ds1, 256)
```

Let's summarize what the above dataflow does:
1. One thread iterates over a shuffled list of (filename, label) pairs, and put them into a queue of size 1000.
2. 25 worker threads take pairs and make them into (preprocessed image, label) pairs.
3. Both 1 and 2 happen in one separate process, and the results are sent back to main process through ZeroMQ.
4. Main process makes batches, and other tensorpack modules will then take care of how they should go into the graph.

Note that in an actual training setup, I used the above multiprocess version for training set since
it's faster to run heavy preprocessing in processes, and use this multithread version only for validation set.

## Sequential Read

Random read may not be a good idea when the data is not on an SSD.
We can also dump the dataset into one single LMDB file and read it sequentially.

```python
from tensorpack.dataflow import *
class BinaryILSVRC12(ILSVRCFiles):
    def get_data(self):
        for fname, label in super(BinaryILSVRC12, self).get_data():
            with open(fname, 'rb') as f:
                jpeg = f.read()
            jpeg = np.asarray(bytearray(jpeg), dtype='uint8')
            yield [jpeg, label]
ds0 = BinaryILSVRC12()
ds1 = PrefetchDataZMQ(ds0, nr_proc=1)
dftools.dump_dataflow_to_lmdb(ds1, '/path/to/ILSVRC-train.lmdb')
```
The above script builds a DataFlow which produces jpeg-encoded ImageNet data.
We store the jpeg string as a numpy array because the function `cv2.imdecode` later expect this format.
Please note we can only use 1 prefetch process to speed up. If `nr_proc>1`, `ds1` will take data
from several forks of `ds0`, then neither the content nor the order of `ds1` will be the same as `ds0`.

It will generate a database file of 140G. We build a DataFlow to read this LMDB file sequentially:
```
ds = LMDBData('/path/to/ILSVRC-train.lmdb', shuffle=False)
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

	  ds = LMDBData('/path/to/ILSVRC-train.lmdb', shuffle=False)
	  ds = LocallyShuffleData(ds, 50000)
	  ds = BatchData(ds, 256, use_list=True)
```
Instead of shuffling all the training data in every epoch (which would require random read),
the added line above maintains a buffer of datapoints and shuffle them once a while.
It will not affect the model as long as the buffer is large enough,
but it can also consume much memory if too large.

Then we add necessary transformations:
```eval_rst
.. code-block:: python
    :emphasize-lines: 3-5

    ds = LMDBData(db, shuffle=False)
    ds = LocallyShuffleData(ds, 50000)
    ds = LMDBDataPoint(ds)
    ds = MapDataComponent(ds, lambda x: cv2.imdecode(x, cv2.IMREAD_COLOR), 0)
    ds = AugmentImageComponent(ds, lots_of_augmentors)
    ds = BatchData(ds, 256)
```

1. `LMDBDataPoint` deserialize the datapoints (from raw bytes to [jpeg_string, label] -- what we dumped in `RawILSVRC12`)
2. Use OpenCV to decode the first component into ndarray
3. Apply augmentations to the ndarray

Both imdecode and the augmentors can be quite slow. We can parallelize them like this:
```eval_rst
.. code-block:: python
    :emphasize-lines: 3,7

    ds = LMDBData(db, shuffle=False)
    ds = LocallyShuffleData(ds, 50000)
    ds = PrefetchData(ds, 5000, 1)
    ds = LMDBDataPoint(ds)
    ds = MapDataComponent(ds, lambda x: cv2.imdecode(x, cv2.IMREAD_COLOR), 0)
    ds = AugmentImageComponent(ds, lots_of_augmentors)
    ds = PrefetchDataZMQ(ds, 25)
    ds = BatchData(ds, 256)
```

Since we are reading the database sequentially, having multiple forked instances of the
base LMDB reader will result in biased data distribution. Therefore we use `PrefetchData` to
launch the base DataFlow in only **one process**, and only parallelize the transformations
with another `PrefetchDataZMQ`
(Nesting two `PrefetchDataZMQ`, however, will result in a different behavior.
These differences are explained in the API documentation in more details.).
Similar to what we did above, you can use `ThreadedMapData` to parallelize as well.

Let me summarize what the above DataFlow does:

1. One process reads LMDB file, shuffle them in a buffer and put them into a `multiprocessing.Queue` (used by `PrefetchData`).
2. 25 processes take items from the queue, decode and process them into [image, label] pairs, and
	 send them through ZMQ IPC pipe.
3. The main process takes data from the pipe, makes batches.

The DataFlow mentioned above (both random read and sequential read) can run at a speed of 1k ~ 2k images per second if you have good CPUs, RAM, disks.
As a reference, tensorpack can train ResNet-18 at 1.2k images/s on 4 old TitanX.
A DGX-1 (8 P100) can train ResNet-50 at 1.7k images/s according to the [official benchmark](https://www.tensorflow.org/performance/benchmarks).
So DataFlow will not be a serious bottleneck if configured properly.

## More Efficient DataFlow

To work with larger datasets (or smaller networks, or more/better GPUs) you could be severely bounded by CPU or disk speed of a single machine.
One way is to optimize the preprocessing routine, for example:

1. Write some preprocessing steps in C++ or use better libraries
2. Move certain preprocessing steps (e.g. mean/std normalization) to TF operators which may be faster
3. Transfer less data, e.g. use uint8 images rather than float32.

Another way to scale is to run DataFlow in a distributed fashion and collect them on the
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
send_dataflow_zmq(df, 'ipc:///tmp/ipc-socket')
```
```python
# Training Machine, training process
df = RemoteDataZMQ('ipc:///tmp/ipc-socket', 'tcp://0.0.0.0:8877')
TestDataSpeed(df).start()
```

[1]: #ref

<div id=ref> </div>

[[1]]. [ImageNet: A Large-Scale Hierarchical Image Database](http://www.image-net.org/papers/imagenet_cvpr09.pdf), CVPR09
