
# Inference

## Inference During Training

There are two ways to do inference during training.

1. The easiest way is to write a callback, and use
  [self.trainer.get_predictor()](../modules/modules/train.html#tensorpack.train.TowerTrainer.get_predictor)
	to get a callable under inference mode.
	See [Write a Callback](extend/callback.html).

2. If your inference follows the paradigm of:
	"fetch some tensors for each input, and aggregate the results".
	You can use the `InferenceRunner` interface with some `Inferencer**.
	This will further support prefetch & data-parallel inference.
	More details to come.

In both methods, your tower function will be called again, with `TowerContext.is_training==False`.
You can use this predicate to choose a different code path in inference mode.

## Inference After Training

Tensorpack is a training interface -- it doesn't care what happened after training.
It saves models to standard checkpoint format.
You can build the graph for inference, load the checkpoint, and then use whatever deployment methods TensorFlow supports.
But you'll need to read TF docs and do it on your own.

### Don't Use Training Metagraph for Inference

Metagraph is the wrong abstraction for a "model". 
It stores the entire graph which contains not only the model, but also all the
training settings (queues, iterators, summaries, evaluations, multi-gpu replications).
Therefore it is usually wrong to import a training metagraph for inference.

It's also very common to change the graph for inference.
For example, you may need a different data layout for CPU inference,
or you may need placeholders in the inference graph (which may not even exist in
the training graph). However metagraph is not designed to be easily modified at all.

To do inference, it's best to recreate a clean graph (and save it if needed).
To construct a new graph, you can simply:
```python
a, b = tf.placeholder(...), tf.placeholder(...)
# call ANY symbolic functions on a, b. e.g.:
with TowerContext('', is_training=False):
	model.build_graph(a, b)
```

### OfflinePredictor
The only tool tensorpack has for after-training inference is [OfflinePredictor](../modules/predict.html#tensorpack.predict.OfflinePredictor),
a simple function to build the graph and return a callable for you.
It is mainly for quick demo purposes.
It only runs inference on numpy arrays, therefore may not be the most efficient way.
Check out examples and docs for its usage.
