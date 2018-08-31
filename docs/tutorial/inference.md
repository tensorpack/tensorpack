
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
You have everything you need for inference or model diagnosis after
training:
1. The trained weights: tensorpack saves them in standard TF checkpoint format.
2. The model: you've already written it yourself with TF symbolic functions.

Therefore, you can build the graph for inference, load the checkpoint, and apply
any processing or deployment TensorFlow supports.
And you'll need to read TF docs and __do it on your own__.

### Don't Use Training Metagraph for Inference

Metagraph is the wrong abstraction for a "model". 
It stores the entire graph which contains not only the mathematical model, but also all the
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
Check out examples and docs for its usage.


OfflinePredictor is only for quick demo purposes.
It runs inference on numpy arrays, therefore may not be the most efficient way.
It also has very limited functionalities.
If you need anything more complicated, please __do it on your own__ because Tensorpack
doesn't care what happened after training.
