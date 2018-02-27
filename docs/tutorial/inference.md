
# Inference

## Inference During Training

There are two ways to do inference during training.

1. The easiest way is to write a callback, and use
  [self.trainer.get_predictor()](../modules/modules/train.html#tensorpack.train.TowerTrainer.get_predictor)
	to get a callable under inference mode.
	See [Write a Callback](extend/callback.html).

2. If your inference follows the paradigm of:
	"fetch some tensors for each input, and aggregate the results".
	You can use the `InferenceRunner` interface with some `Inferencer`.
	This will further support prefetch & data-parallel inference.
	More details to come.

In both methods, your tower function will be called again, with `TowerContext.is_training==False`.
You can build a different graph using this predicate.

## Inference After Training

Tensorpack doesn't care what happened after training.
It saves models to standard checkpoint format, plus a metagraph protobuf file.
They are sufficient to use with whatever deployment methods TensorFlow supports.
But you'll need to read TF docs and do it on your own.

Please note that, the metagraph saved during training is the training graph.
But sometimes you need a different one for inference.
For example, you may need a different data layout for CPU inference,
or you may need placeholders in the inference graph, or the training graph contains multi-GPU replication
which you want to remove. In fact, directly import a huge training metagraph is usually not a good idea for deployment.

In this case, you can always construct a new graph by simply:
```python
a, b = tf.placeholder(...), tf.placeholder(...)
# call symbolic functions on a, b
```

The only tool tensorpack has for after-training inference is [OfflinePredictor](../modules/predict.html#tensorpack.predict.OfflinePredictor),
a simple function to build the graph and return a callable for you.
It is mainly for quick demo purposes.
It only runs inference on numpy arrays, therefore may not be the most efficient way.
Check out examples and docs for its usage.
