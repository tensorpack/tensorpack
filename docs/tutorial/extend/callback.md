
## Write a Callback

__Everything__ other than the training iterations happen in the callbacks.
Most of the fancy things you want to do will probably end up here.

Callbacks are called during training.
The time where each callback method gets called is demonstrated in this snippet.
```python
def train(self):
  # ... a predefined trainer may create graph for the model here ...
  callbacks.setup_graph()
  # ... create session, initialize session, finalize graph ...
  # start training:
  with sess.as_default():
    callbacks.before_train()
    for epoch in range(starting_epoch, max_epoch + 1):
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
You can overwrite any of the following methods in the new callback:

* `_setup_graph(self)`

  Create any tensors/ops in the graph which you might need to use in the callback.
  This method exists to fully separate between "define" and "run", and also to
  avoid the common mistake to create ops inside
  loops. All changes to the graph should be made in this method.

  To access tensors/ops which are already defined,
  you can use TF methods such as
  [`graph.get_tensor_by_name`](https://www.tensorflow.org/api_docs/python/tf/Graph#get_tensor_by_name).
  If you're using a `TowerTrainer`, more tools are available:

  - Use `self.trainer.tower_func.towers` to access the
  	[tower handles](../../modules/tfutils.html#tensorpack.tfutils.tower.TowerTensorHandles),
  	and therefore the tensors in each tower.
  - [self.get_tensors_maybe_in_tower()](../../modules/callbacks.html#tensorpack.callbacks.Callback.get_tensors_maybe_in_tower)
  	is a helper function to look for tensors first globally, then in the first training tower.
  - [self.trainer.get_predictor()](../../modules/train.html#tensorpack.train.TowerTrainer.get_predictor)
  	is a helper function to create a callable under inference mode.

* `_before_train(self)`

  Can be used to run some manual initialization of variables, or start some services for the training.

* `_after_train(self)`

  Usually some finalization work.

* `_before_epoch(self)`, `_after_epoch(self)`

  `_trigger_epoch` should be enough for most cases, as can be seen from the scheduling snippet above.
  These two methods should be used __only__ when you really need something to happen __immediately__ before/after an epoch.
	And when you do need to use them, make sure they are very very fast to avoid affecting other callbacks which use them.

* `_before_run(self, ctx)`, `_after_run(self, ctx, values)`

  These are the equivalence of [tf.train.SessionRunHook](https://www.tensorflow.org/api_docs/python/tf/train/SessionRunHook).
  Please refer to TensorFlow documentation for detailed API.
  They are used to run extra ops / eval extra tensors / feed extra values __along with__ the actual training iterations.

  __IMPORTANT__ Note the difference between running __along with__ an iteration and running __after__ an iteration.
  When you write

  ```python
  def _before_run(self, _):
    return tf.train.SessionRunArgs(fetches=my_op)
  ```

  The training loops would become equivalent to `sess.run([training_op, my_op])`.
  
  However, if you write `my_op.run()` in `_trigger_step`, the training loop would become
  `sess.run(training_op); sess.run(my_op);`.
  Usually the difference matters, please choose carefully.
  
  If you want to run ops that depend on your inputs, it's better to run it
  __along with__ the training iteration, to avoid wasting a datapoint and avoid
  messing up hooks of the `InputSource`.

* `_trigger_step(self)`

  Do something (e.g. running ops, print stuff) after each step has finished.
  Be careful to only do light work here because it could affect training speed.

* `_trigger_epoch(self)`

  Do something after each epoch has finished. This method calls `self.trigger()` by default.

* `_trigger(self)`

  Define something to do here without knowing how often it will get called.
  By default it will get called by `_trigger_epoch`,
  but you can customize the scheduling of this method by
  [`PeriodicTrigger`](../../modules/callbacks.html#tensorpack.callbacks.PeriodicTrigger),
  to let this method run every k steps or every k epochs.

### What you can do in the callback

* Access tensors / ops (details mentioned above):
	* For existing tensors/ops created in the tower, access them through [self.trainer.towers](../../modules/train.html#tensorpack.train.TowerTrainer.towers).
	* Extra tensors/ops have to be created in `_setup_graph` callback method.
* Access the current graph and session by `self.trainer.graph` and
  `self.trainer.sess`, `self.trainer.hooked_sess`.
  Note that calling `(hooked_)sess.run` to evaluate tensors may have unexpected
  effect in certain scenarios. 
  In general, use `sess.run` to evaluate tensors that do not depend on the inputs.
  And use `_{before,after}_run` to evaluate tensors together with inputs if the
  tensors depend on the inputs.
* Write stuff to the monitor backend, by `self.trainer.monitors.put_xxx`.
  The monitors might direct your events to TensorFlow events file, JSON file, stdout, etc.
  You can access history monitor data as well. See the docs for [Monitors](../../modules/callbacks.html#tensorpack.callbacks.Monitors)
* Access the current status of training, such as `self.epoch_num`, `self.global_step`. See docs of [Callback](../../modules/callbacks.html#tensorpack.callbacks.Callback)
* Stop training by `raise StopTraining()` (with `from tensorpack.train import StopTraining`).
* Anything else that can be done with plain python.

### Typical Steps about Writing/Using a Callback

* Define the callback in `__init__`, prepare for it in `_setup_graph, _before_train`.
* Know whether you want to do something __along with__ the training iterations or not.
  If yes, implement the logic with `_{before,after}_run`.
  Otherwise, implement in `_trigger`, or `_trigger_step`.
* You can choose to only implement "what to do", and leave "when to do" to
  other wrappers such as
  [PeriodicTrigger](../../modules/callbacks.html#tensorpack.callbacks.PeriodicTrigger),
  [PeriodicCallback](../../modules/callbacks.html#tensorpack.callbacks.PeriodicCallback),
  or [EnableCallbackIf](../../modules/callbacks.html#tensorpack.callbacks.EnableCallbackIf).
	Of course you also have the freedom to implement "what to do" and "when to do" altogether.


### Examples

Check source code of the [existing tensorpack callbacks](../../modules/callbacks.html). 
Or grep 'Callback' in tensorpack examples for those implemented as extensions.
