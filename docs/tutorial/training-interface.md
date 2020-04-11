
# Training Interface

Tensorpack trainers have a verbose interface for maximum flexibility.
Then, there are interfaces built on top of trainers to simplify the use,
when you don't want to customize too much.

### With ModelDesc and TrainConfig

This interface is enough for most types of single-cost tasks.
A lot of examples are written in this interface.

[SingleCost trainers](../modules/train.html#tensorpack.train.SingleCostTrainer)
expects 4 arguments to setup the graph: input signatures, `InputSource`, get_cost function, and an optimizer.
`ModelDesc` describes a model by packing 3 of them together into one object:

```python
class MyModel(ModelDesc):
  def inputs(self):
    return [tf.TensorSpec(shape, dtype, name), tf.TensorSpec(shape, dtype, name), ... ]

  def build_graph(self, tensorA, tensorB, ...):  # inputs
    # build the graph
    return cost   # returns the cost tensor

  def optimizer(self):
    return tf.train.GradientDescentOptimizer(0.1)
```

`inputs()` should define the metainfo of all the inputs your graph will take to build.

`build_graph()` takes inputs tensors that matches what you've defined in `inputs()`.

You can use any symbolic functions in `build_graph`, including TensorFlow core library
functions and other symbolic libraries.
`build_graph` will be the tower function, so you need to follow [some rules](trainer.html#tower-trainer).
Because this interface is specialized for single-cost training, you need to return the cost tensor.

After defining such a model, use it with `TrainConfig` and `launch_train_with_config`:

```python
config = TrainConfig(
   model=MyModel()
   dataflow=my_dataflow,
   # data=my_inputsource, # alternatively, use an InputSource
   callbacks=[...],    # some default callbacks are automatically applied
   # some default monitors are automatically applied
   steps_per_epoch=300,   # default to the size of your InputSource/DataFlow
)

trainer = SomeTrainer()
# trainer = SyncMultiGPUTrainerParameterServer(8)
launch_train_with_config(config, trainer)
```
See the docs of
[TrainConfig](../modules/train.html#tensorpack.train.TrainConfig)
and
[launch_train_with_config](../modules/train.html#tensorpack.train.launch_train_with_config)
for detailed functionalities.

The function `launch_train_with_config(config, trainer)`
uses the raw trainer interface under the hood, and is almost equivalent to the following two lines of code:
```python
trainer.setup_graph(
    my_model.get_input_signature(),
    my_input_source,  # or QueueInput(my_dataflow)
    my_model.build_graph,
    my_model.get_optimizer)
trainer.train_with_defaults(
    callbacks=config.callbacks,
    monitors=config.monitors,
    session_creator=config.session_creator,
    session_init=config.session_init,
    steps_per_epoch=config.steps_per_epoch,
    starting_epoch=config.starting_epoch,
    max_epoch=config.max_epoch,
    extra_callbacks=config.extra_callbacks)
```
If you need more control (e.g., if you want to construct the callbacks after
setting up the graph), you can write the above two lines by yourself instead.
You don't need to construct a `TrainConfig` any more in that case.
The function `launch_train_with_config` exists mainly for historical reasons.

### Keras Interface

Some wrappers were made on top of tensorpack trainers, to create a Keras-like interface.
See the experimental [Tensorpack+Keras examples](../../examples/keras) for details.

### Raw Trainer Interface

To get a lower-level control, you can also access trainer methods directly:

__Build the graph__:
For single-cost trainers, build the graph by calling
[SingleCostTrainer.setup_graph](../modules/train.html#tensorpack.train.SingleCostTrainer.setup_graph).
For other types of tasks, you can build the graph by yourself.

__Start training__: Call
[Trainer.train()](../modules/train.html#tensorpack.train.Trainer.train) to start
training, or call
[Trainer.train_with_defaults()](../modules/train.html#tensorpack.train.Trainer.train_with_defaults)
which applies some defaults options for common use cases.

Read their API documentation and the
[advanced trainer tutorial](./extend/trainer.md)
for more details.
