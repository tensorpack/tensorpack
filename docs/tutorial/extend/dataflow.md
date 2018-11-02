
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
    # load data from somewhere with Python
    yield [my_array, my_label]

dataflow = DataFromGenerator(my_data_loader)
```

To write more complicated DataFlow, you need to inherit the base `DataFlow` class.
Usually, you just need to implement the `__iter__()` method which yields a datapoint every time.
```python
class MyDataFlow(DataFlow):
  def __iter__(self):
    for k in range(100):
      digit = np.random.rand(28, 28)
      label = np.random.randint(10)
      yield [digit, label]
      
df = MyDataFlow()
df.reset_state()
for datapoint in df:
    print(datapoint[0], datapoint[1])
```

Optionally, you can implement the `__len__` and `reset_state` method. 
The detailed semantics of these three methods are explained 
in the [API documentation](../../modules/dataflow.html#tensorpack.dataflow.DataFlow).

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

  def __iter__(self):
    for datapoint in self.ds.get_data():
      # do something
      yield new_datapoint
```

Some built-in dataflows, e.g.
[MapData](../../modules/dataflow.html#tensorpack.dataflow.MapData) and 
[MapDataComponent](../../modules/dataflow.html#tensorpack.dataflow.MapDataComponent)
can do common types of data processing for you.
