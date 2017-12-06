
# Inference

## Inference During Training

There are two ways to do inference during training.

1. The easiest way is to write a callback, and use
  `self.trainer.get_predictor()` to get a callable under inference mode.
	See [Write a Callback](extend/callback.html).

2. If your inference follows the paradigm of:
	"fetch some tensors for each input, and aggregate the results".
	You can use the `InferenceRunner` interface with some `Inferencer`.
	This will further support prefetch & data-parallel inference.
	More details to come.


## Inference After Training

Tensorpack doesn't care what happened after training.
It saves models to standard checkpoint format, plus a metagraph protobuf file.
They are sufficient to use with whatever deployment methods TensorFlow supports.
But you'll need to read TF docs and do it on your own.

Please note that, the metagraph saved during training is the training graph.
But sometimes you need a different one for inference.
For example, you may need a different data layout for CPU inference,
or you may need placeholders in the inference graph, or the training graph contains multi-GPU replication
which you want to remove.
In this case, you can always create a new graph with pure TensorFlow.

The only thing tensorpack has for this purpose is `OfflinePredictor`,
a simple function to build the graph and a callable for you.
It is mainly for quick demo purpose.
It only runs inference on Python data, therefore may not be the most efficient way.
Check out some examples for its usage.
