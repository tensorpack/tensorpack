
## Write a Callback

The time where each callback method gets called is demonstrated in this snippet.
```python
def train(self):
  # ... a predefined trainer may create graph for the model here ...
  callbacks.setup_graph()
  # ... create session, initialize session, finalize graph ...
  # start training:
  with sess.as_default():
    callbacks.before_train()
    for epoch in range(epoch_start, epoch_end):
      callbacks.before_epoch()
      for step in range(steps_per_epoch):
        self.run_step()  # callbacks.{before,after}_run are hooked with session
        callbacks.trigger_step()
      callbacks.after_epoch()
      callbacks.trigger_epoch()
    callbacks.after_train()
```
Note that at each place, each callback will be called in the order they are given to the trainer.


### Explain the Callback Methods

To write a callback, subclass `Callback` and implement the corresponding underscore-prefixed methods.
You can overwrite any of the following methods to define a new callback:

* `_setup_graph(self)`

  Create any ops / tensors in the graph which you might need to use in the callback.
  This method is to separate between "define" and "run", and also to
  avoid the common mistake to create ops inside
  loops. All changes to the graph should be made in this method.

  To access ops which are already defined,
  you can use TF methods such as
  [`graph.get_tensor_by_name`](https://www.tensorflow.org/api_docs/python/tf/Graph#get_tensor_by_name).
  If you're using a `TowerTrainer` instance, more tools are available:

  - Use `self.trainer.tower_func.towers` to access the
  	[tower handles](../../modules/tfutils.html#tensorpack.tfutils.tower.TowerTensorHandles),
  	and therefore the tensors in each tower.
  - [self.get_tensors_maybe_in_tower()](../../modules/callbacks.html#tensorpack.callbacks.Callback.get_tensors_maybe_in_tower)
  	is a helper function to access tensors in the first training tower.
  - [self.trainer.get_predictor()](../../modules/train.html#tensorpack.train.TowerTrainer.get_predictor)
  	is a helper function to create a callable under inference mode.

* `_before_train(self)`

  Can be used to run some manual initialization of variables, or start some services for the training.

* `_after_train(self)`

  Usually some finalization work.

* `_before_epoch(self)`, `_after_epoch(self)`

  Use them __only__ when you really need something to happen __immediately__ before/after an epoch.
  Otherwise, `_trigger_epoch` should be enough.

* `_before_run(self, ctx)`, `_after_run(self, ctx, values)`

  These are the equivalence of [tf.train.SessionRunHook](https://www.tensorflow.org/api_docs/python/tf/train/SessionRunHook).
  Please refer to TensorFlow documentation for detailed API.
  They are used to run extra ops / eval extra tensors / feed extra values __along with__ the actual training iterations.

  Note the difference between running __along with__ an iteration and running after an iteration.
  When you write

  ```python
  def _before_run(self, _):
    return tf.train.SessionRunArgs(fetches=my_op)
  ```

  The training loops would become `sess.run([training_op, my_op])`.
  This is different from `sess.run(training_op); sess.run(my_op);`,
  which is what you would get if you run the op in `_trigger_step`.

* `_trigger_step(self)`

  Do something (e.g. running ops, print stuff) after each step has finished.
  Be careful to only do light work here because it could affect training speed.

* `_trigger_epoch(self)`

  Do something after each epoch has finished. Will call `self.trigger()` by default.

* `_trigger(self)`

  Define something to do here without knowing how often it will get called.
  By default it will get called by `_trigger_epoch`,
  but you can customize the scheduling of this method by
  [`PeriodicTrigger`](../../modules/callbacks.html#tensorpack.callbacks.PeriodicTrigger),
  to let this method run every k steps or every k epochs.

### What you can do in the callback

* Access tensors / ops in either training / inference mode (need to create them in `_setup_graph`).
* Write stuff to the monitor backend, by `self.trainer.monitors.put_xxx`.
  The monitors might direct your events to TensorFlow events file, JSON file, stdout, etc.
  You can get history monitor data as well. See the docs for [Monitors](../../modules/callbacks.html#tensorpack.callbacks.Monitors)
* Access the current status of training, such as `epoch_num`, `global_step`. See [here](../../modules/callbacks.html#tensorpack.callbacks.Callback)
* Stop training by `raise StopTraining()` (with `from tensorpack.train import StopTraining`).
* Anything else that can be done with plain python.

### Typical Steps about Writing/Using a Callback

* Define the callback in `__init__`, prepare for it in `_setup_graph, _before_train`.
* Know whether you want to do something __along with__ the session run or not.
  If yes, implement the logic with `_{before,after}_run`.
  Otherwise, implement in `_trigger`, or `_trigger_step`.
* You can choose to only implement "what to do", and leave "when to do" to
  other wrappers such as
  [PeriodicTrigger](../../modules/callbacks.html#tensorpack.callbacks.PeriodicTrigger),
  [PeriodicRunHooks](../../modules/callbacks.html#tensorpack.callbacks.PeriodicRunHooks),
  or [EnableCallbackIf](../../modules/callbacks.html#tensorpack.callbacks.EnableCallbackIf).

