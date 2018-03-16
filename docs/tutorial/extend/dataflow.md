
### Write a DataFlow

#### Write a Source DataFlow

There are several existing DataFlow, e.g. [ImageFromFile](../../modules/dataflow.html#tensorpack.dataflow.ImageFromFile),
[DataFromList](../../modules/dataflow.html#tensorpack.dataflow.DataFromList),
which you can use if your data format is simple.
In general, you probably need to write a source DataFlow to produce data for your task,
and then compose it with existing modules (e.g. mapping, batching, prefetching, ...).

The easiest way to create a DataFlow to load custom data, is to wrap a custom generator, e.g.:
```python
def my_data_loader():
  while True:
    # load data from somewhere
    yield [my_array, my_label]

dataflow = DataFromGenerator(my_data_loader)
```

To write more complicated DataFlow, you need to inherit the base `DataFlow` class.
Usually, you just need to implement the `get_data()` method which yields a datapoint every time.
```python
class MyDataFlow(DataFlow):
  def get_data(self):
    for k in range(100):
      digit = np.random.rand(28, 28)
      label = np.random.randint(10)
      yield [digit, label]
```

Optionally, you can implement the following two methods:

+ `size()`. Return the number of elements the generator can produce. Certain tensorpack features might use it.

+ `reset_state()`. It is guaranteed that the actual process which runs a DataFlow will invoke this method before using it.
  So if this DataFlow needs to do something after a `fork()`, you should put it here.
  `reset_state()` must be called once and only once for each DataFlow instance.

  A typical example is when your DataFlow uses random number generator (RNG). Then you would need to reset the RNG here.
  Otherwise, child processes will have the same random seed. The `RNGDataFlow` base class does this for you.
  You can subclass `RNGDataFlow` to access `self.rng` whose seed has been taken care of.

DataFlow implementations for several well-known datasets are provided in the
[dataflow.dataset](../../modules/dataflow.dataset.html)
module, you can take them as a reference.

#### More Data Processing

You can put any data processing you need in the source DataFlow you write, or you can write a new DataFlow for data
processing on top of the source DataFlow, e.g.:

```python
class ProcessingDataFlow(DataFlow):
  def __init__(self, ds):
    self.ds = ds

  def get_data(self):
    for datapoint in self.ds.get_data():
      # do something
      yield new_datapoint
```
