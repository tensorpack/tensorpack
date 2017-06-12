
## Write a callback

The places where each callback gets called is demonstrated in this snippet:

```python
def main_loop():
  # create graph for the model
  callbacks.setup_graph()
  # create session, initialize session, finalize graph ...
  # start training:
  callbacks.before_train()
  for epoch in range(epoch_start, epoch_end):
		callbacks.before_epoch()
    for step in range(steps_per_epoch):
      run_step()  # callbacks.{before,after}_run are hooked with session
      callbacks.trigger_step()
		callbacks.after_epoch()
    callbacks.trigger_epoch()
  callbacks.after_train()
```

You can overwrite any of the following methods to define a new callback:

* `_setup_graph(self)`

To separate between "define" and "run", and also to avoid the common mistake to create ops inside
loops, all changes to the graph should be made in this method. No session has been created at this time.

TODO how to access the tensors already defined.

* `_before_train(self)`

Can be used to run some manual initialization of variables, or start some services for the whole training.

* `_trigger_step(self)`

Do something (including running ops) after each step has finished.
Be careful to only do light work here because it could affect training speed.

* `_before_run(self, ctx)`, `_after_run(self, ctx, values)`

This two are the equivlent of [tf.train.SessionRunHook](https://www.tensorflow.org/api_docs/python/tf/train/SessionRunHook).
Please refer to TensorFlow documentation for detailed API.
They are used to run extra ops / eval extra tensors / feed extra values __along with__ the actual training iteration.

Note the difference between running __along with__ an iteration and running after an iteration.
When you write

```python
def _before_run(self, _):
  return tf.train.SessionRunArgs(fetches=my_op)
```

The training loops would become `sess.run([training_op, my_op])`.
This is different from `sess.run(training_op); sess.run(my_op);`,
which is what you would get if you run the op in `_trigger_step`.

* `_trigger_epoch(self)`

Do something after each epoch has finished. Will call `self.trigger()` by default.

* `_trigger(self)`

By default will get called by `_trigger_epoch`,
but you can then customize the scheduling of this callback by
`PeriodicTrigger`, to let this method run every k steps or every k epochs.

* `_after_train(self)`

Do some finalization work.
