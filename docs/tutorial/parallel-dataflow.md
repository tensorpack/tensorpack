# Parallel DataFlow

This tutorial explains the parallel building blocks
inside DataFlow, since most of the time they are the only things
needed to build an efficient dataflow.


## Concepts: how to make things parallel:

Code does not automatically utilize multiple CPUs.
You need to specify how to split the tasks across CPUs.
A tensorpack DataFlow can be parallelized across CPUs in the following two ways:

### Run Multiple Identical DataFlows

In this pattern, multiple identical DataFlows run on multiple CPUs,
and put results in a queue.
The master receives the output from the queue.

To use this pattern with multi-processing, you can do:
```
d1 = MyDataFlow()   # some dataflow written by the user
d2 = MultiProcessRunnerZMQ(d1, num_proc=20)
```

The second line starts 20 processes running `d1`, and merge the results.
You can then obtain the results in `d2`.

Note that, all the workers run independently in this pattern.
This means you need to have sufficient randomness in `d1`.
If `d1` produce the same sequence in each worker,
then `d2` will produce repetitive data points.

There are some other similar issues you need to take care of when using this pattern.
You can find them at the
[API documentation](../modules/dataflow.html#tensorpack.dataflow.MultiProcessRunnerZMQ).


### Distribute Tasks to Multiple Workers

In this pattern, the master sends datapoints (the tasks)
to multiple workers.
The workers are responsible for executing a (possibly expensive) mapping
function on the datapoints, and send the results back to the master.
An example with multi-processing is like this:

```
d1 = MyDataFlow()   # a dataflow that produces [image file name, label]

def f(file_name, label):
    # read image
    # run heavy pre-proecssing / augmentation on the image
    return img, label

d2 = MultiProcessMapData(dp, num_proc=20, f)
```

The main difference between this pattern and the first, is that:
1. `d1` is not executed in parallel. Only `f` runs in parallel.
  Therefore you don't have to worry about randomness or data distribution shift.
  But you need to make `d1` very efficient (e.g. let it produce small metadata).
2. More communication is required, because it needs to send data to workers.

See its [API documentation](../modules/dataflow.html#tensorpack.dataflow.MultiProcessMapData)
to learn more details.

## Threads & Processes

Both the above two patterns can be used with
__either multi-threading or multi-processing__, with the following builtin DataFlows:

* [MultiProcessRunnerZMQ](../modules/dataflow.html#tensorpack.dataflow.MultiProcessRunnerZMQ)
  or [MultiProcessRunner](../modules/dataflow.html#tensorpack.dataflow.MultiProcessRunner)
* [MultiThreadRunner](../modules/dataflow.html#tensorpack.dataflow.MultiThreadRunner)
* [MultiProcessMapDataZMQ](../modules/dataflow.html#tensorpack.dataflow.MultiProcessMapDataZMQ)
* [MultiThreadMapData](../modules/dataflow.html#tensorpack.dataflow.MultiThreadMapData)

(ZMQ means [ZeroMQ](http://zeromq.org/), a communication library that is more
efficient than Python's multiprocessing module)

Using threads and processes have their pros and cons:

1. Threads in Python are limted by [GIL](https://wiki.python.org/moin/GlobalInterpreterLock).
   Threads in one process cannot interpret Python statements in parallel.
   As a result, multi-threading may not scale well, if the workers spend a
   significant amount of time in the Python interpreter.
2. Processes need to pay the overhead of communication with each other.

Though __processes are most commonly used__,
The best choice of the above parallel utilities varies across machines and tasks.
You can even combine threads and processes sometimes.

Note that in tensorpack, all the multiprocessing DataFlow with "ZMQ" in the name creates
__zero Python threads__: this is a key implementation detail that makes tensorpack DataFlow
faster than the alternatives in Keras or PyTorch.

For a new task, you often need to do a quick benchmark to choose the best pattern.
See [Performance Tuning Tutorial](./performance-tuning.md)
on how to effectively understand the performance of a DataFlow.

See also [Efficient DataFlow](./efficient-dataflow.md)
for real examples using the above DataFlows.

