
# Inference

## Inference During Training

There are two ways to do inference during training.

1. The easiest way is to write a callback, and use
  [self.trainer.get_predictor()](../modules/modules/train.html#tensorpack.train.TowerTrainer.get_predictor)
	to get a callable under inference mode.
	See [Write a Callback](extend/callback.html).

2. If your inference follows the paradigm of:
	"evaluate some tensors for each input, and aggregate the results in the end".
	You can use the `InferenceRunner` interface with some `Inferencer**.
	This will further support prefetch & data-parallel inference.
	More details to come.

In both methods, your tower function will be called again, with `TowerContext.is_training==False`.
You can use this predicate to choose a different code path in inference mode.

## Inference After Training

Tensorpack is a training interface -- __it doesn't care what happened after training__.
You already have everything you need for inference or model diagnosis after
training:
1. The model (the graph): you've already written it yourself with TF symbolic functions.
2. The trained parameters: tensorpack saves them in standard TF checkpoint format.

Therefore, you can build the graph for inference, load the checkpoint, and apply
any processing or deployment TensorFlow supports.
These are unrelated to tensorpack, and you'll need to read TF docs and __do it on your own__.

### Step 1: build the model

You can build a graph however you like, with pure TensorFlow. If your model is written with
tensorpack's `ModelDesc`, you can also build it like this:

```python
a, b = tf.placeholder(...), tf.placeholder(...)
# call ANY symbolic functions on a, b. e.g.:
with TowerContext('', is_training=False):
	model.build_graph(a, b)
```

```eval_rst
.. note:: **Do not use metagraph for inference!**. 

	Metagraph is the wrong abstraction for a "model". 
	It stores the entire graph which contains not only the mathematical model, but also all the
	training settings (queues, iterators, summaries, evaluations, multi-gpu replications).
	Therefore it is usually wrong to import a training metagraph for inference.

	It's also very common to change the graph for inference.
	For example, you may need a different data layout for CPU inference,
	or you may need placeholders in the inference graph (which may not even exist in
	the training graph). However metagraph is not designed to be easily modified at all.

	To do inference, it's best to recreate a clean graph (and save it if needed) by yourself.
```

### Step 2: load the checkpoint

You can just use `tf.train.Saver` for all the work.
Alternatively, use tensorpack's `SaverRestore(path).init(tf.get_default_session())`


### OfflinePredictor

Tensorpack provides one tool [OfflinePredictor](../modules/predict.html#tensorpack.predict.OfflinePredictor),
to merge the above two steps together.
It has simple functionailities to build the graph, load the checkpoint, and return a callable for you.
Check out examples and docs for its usage.

OfflinePredictor is only for quick demo purposes.
It runs inference on numpy arrays, therefore may not be the most efficient way.
It also has very limited functionalities.
If you need anything more complicated, please __do it on your own__ because Tensorpack
doesn't care what happened after training.
