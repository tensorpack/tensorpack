
### Write a DataFlow

There are several existing DataFlow, e.g. [ImageFromFile](../../modules/dataflow.html#tensorpack.dataflow.ImageFromFile),
[DataFromList](../../http://tensorpack.readthedocs.io/en/latest/modules/dataflow.html#tensorpack.dataflow.DataFromList),
which you can use if your data format is simple.
In general, you probably need to write a source DataFlow to produce data for your task,
and then compose it with existing modules (e.g. mapping, batching, prefetching, ...).

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

	A typical situation is when your DataFlow uses random number generator (RNG). Then you would need to reset the RNG here.
	Otherwise, child processes will have the same random seed. The `RNGDataFlow` base class does this for you.
	You can subclass `RNGDataFlow` to access `self.rng` whose seed has been taken care of.

The convention is that, `reset_state()` must be called once and usually only once for each DataFlow instance.
To reinitialize the dataflow (i.e. get a new iterator from the beginning), simply call `get_data()` again.

DataFlow implementations for several well-known datasets are provided in the
[dataflow.dataset](../../modules/dataflow.dataset.html)
module, you can take them as a reference.
