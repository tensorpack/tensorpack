
# Callbacks

Apart from the actual training iterations that minimize the cost,
you almost surely would like to do something else.
Callbacks are such an interface to describe what to do besides the
training iterations.

There are several places where you might want to do something else:

* Before the training has started (e.g. initialize the saver, dump the graph)
* Along with each training iteration (e.g. run some other operations in the graph)
* Between training iterations (e.g. update the progress bar, update hyperparameters)
* Between epochs (e.g. save the model, run some validation)
* After the training (e.g. send the model somewhere, send a message to your phone)

We found people traditionally tend to write the training loop together with these extra features.
This makes the loop lengthy, and the code for the same feature probably get separated.
By writing callbacks to implement what to do at each place, tensorpack trainers
will call the callbacks at the proper time.
Therefore the code can be reused with one single line, as long as you are using tensorpack trainers.

For example, these are the callbacks I used when training a ResNet:

```python
TrainConfig(
  # ...
  callbacks=[
    # save the model every epoch
    ModelSaver(),
    # backup the model with best validation error
    MinSaver('val-error-top1'),
    # run inference on another Dataflow every epoch, compute top1/top5 classification error and save them in log
    InferenceRunner(dataset_val, [
        ClassificationError('wrong-top1', 'val-error-top1'),
        ClassificationError('wrong-top5', 'val-error-top5')]),
    # schedule the learning rate based on epoch number
    ScheduledHyperParamSetter('learning_rate',
                              [(30, 1e-2), (60, 1e-3), (85, 1e-4), (95, 1e-5)]),
    # can manually set the learning rate during training
    HumanHyperParamSetter('learning_rate'),
    # send validation error to my phone through pushbullet
    SendStat('curl -u your_id_xxx: https://api.pushbullet.com/v2/pushes \\
               -d type=note -d title="validation error" \\
               -d body={val-error-top1} > /dev/null 2>&1',
               'val-error-top1'),
    # record GPU utilizations during training
    GPUUtilizationTracker(),
    # can pause the training and start a debug shell, to observe what's going on
    InjectShell(shell='ipython')
  ],
  extra_callbacks=[    # these callbacks are enabled by default already
    # maintain and summarize moving average of some tensors defined in the model (e.g. training loss, training error)
    MovingAverageSummary(),
    # draw a nice progress bar
    ProgressBar(),
    # run `tf.summary.merge_all` every epoch and send results to monitors
    MergeAllSummaries(),
    # run ops in GraphKeys.UPDATE_OPS collection along with training, if any
    RunUpdateOps(),
  ],
  monitors=[        # monitors are a special kind of callbacks. these are also enabled by default
    # write everything to tensorboard
    TFEventWriter(),
    # write all scalar data to a json file, for easy parsing
    JSONWriter(),
    # print all scalar data every epoch (can be configured differently)
    ScalarPrinter(),
  ]
)
```

Notice that callbacks cover every detail of training, ranging from graph operations to the progress bar.
This means you can customize every part of the training to your preference, e.g. display something
different in the progress bar, evaluating part of the summaries at a different frequency, etc.
These features may not be always useful, but think about how messy the main loop would look like if you
were to write the logic together with the loops, and how easy your life will be if you could enable
these features with one line when you need them.

See [Write a callback](http://tensorpack.readthedocs.io/en/latest/tutorial/extend/callback.html)
for details on how callbacks work, what they can do, and how to write them.
