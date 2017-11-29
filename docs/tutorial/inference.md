
# Inference

## Inference During Training

There are two ways to do inference during training.

1. The easiest way is to write a callback, and use
`  self.trainer.get_predictor()` to get a callable under inference mode.
	See [Write a Callback](extend/callback.html)

2. If your inference follows the paradigm of:
	"fetch some tensors for each input, and aggregate the results".
	You can use the `InferenceRunner` interface with some `Inferencer`.
	This will further support prefetch & data-parallel inference.
	More details to come.


## Inference After Training

Tensorpack doesn't care what happened after training.
It saves models to standard checkpoint format, plus a metagraph protobuf file.
They are sufficient to use with whatever deployment methods TensorFlow supports.
But you'll need to read the docs and do it on your own.

The only thing tensorpack has is `OfflinePredictor`,
a simple function to build the graph and a callable for you.
It only runs inference on Python data, therefore may not be the best way.
It is mainly for quick demo purpose.
Check out some examples for the usage.
