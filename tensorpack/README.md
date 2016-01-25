
## Tensorpack High-Level Tutorial

### How to feed data
Define a `DataFlow` instance to feed data.
See [Dataflow documentation](https://github.com/ppwwyyxx/tensorpack/tree/master/tensorpack/dataflow)

### How to define a model
Take a look at the `get_model` function in [mnist example](https://github.com/ppwwyyxx/tensorpack/blob/master/example_mnist.py) first.

To define a model, write a `get_model` function which accepts two arguments:
+ inputs: a list of variables used as input in training. inputs could be batched or not batched (see
	[training](#how-to-perform-training))
+ is_training: the graph for training and inference could be different (e.g. dropout, augmentation),
	`get_model` function should use this variable to know is it doing training or inference.

The function should define a graph based on input variables.
It could use any pre-defined routines in [tensorpack/models](https://github.com/ppwwyyxx/tensorpack/tree/master/tensorpack/models),
or use tensorflow symbolic functions.

It may also define other helper variables to monitor the training,
(e.g. accuracy), and add tensorboard summaries you need. (See [howto summary](#use-tensorboard-summary))

Also, it's helpful to give names to some important variables used in inference. (See
[inference](#how-to-perform-inference)).

The function should at last return the cost to minimize.

### How to perform training


### How to perform inference


### How to add new models

### Use tensorboard summary
 <!--
    -	what will be automatically summaried
		-->
