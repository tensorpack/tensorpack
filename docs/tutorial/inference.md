
# Inference

## Inference During Training

There are two ways to do inference during training.

1. The easiest way is to write a callback, and use
  [self.trainer.get_predictor()](../modules/modules/train.html#tensorpack.train.TowerTrainer.get_predictor)
	to get a callable under inference mode.
	See [Write a Callback](extend/callback.html).

2. If your inference follows the paradigm of:
	"evaluate some tensors for each input, and aggregate the results in the end".
	You can use the `InferenceRunner` interface with some `Inferencer`.
	This will further support prefetch & data-parallel inference.
	
    Currently this lacks documentation, but you can refer to examples
    that uses `InferenceRunner` or custom `Inferencer` to learn more.

In both methods, your tower function will be called again, with `TowerContext.is_training==False`.
You can use this predicate to choose a different code path in inference mode.


## Inference After Training: What Tensorpack Does

Tensorpack provides some small tools to do the most basic types of inference for demo purposes.
You can use them but
__these approaches are often suboptimal and may fail__.
They may often be inefficient or lack functionalities you need.

If you need anything more complicated, please
learn what TensorFlow can do, and __do it on your own__ because Tensorpack
is a training interface and doesn't focus on what happened after training.

### OfflinePredictor

Tensorpack provides  [OfflinePredictor](../modules/predict.html#tensorpack.predict.OfflinePredictor),
for inference demo after training.
It has functionailities to build the graph, load the checkpoint, and
return a callable for you for simple prediction. Refer to its docs for details.

OfflinePredictor is only for quick demo purposes.
It runs inference on numpy arrays, therefore may not be the most efficient way.
It also has very limited functionalities.

A simple example of how it works:
```python
pred_config = PredictConfig(
    session_init=get_model_loader(model_path),
    model=YourModel(),
    input_names=['input1', 'input2'],  # tensor names in the graph, or name of the declared inputs
    output_names=['output1', 'output2'])  # tensor names in the graph
predictor = OfflinePredictor(pred_config)
output1_array, output2_array = predictor(input1_array, input2_array)
```

It's __common to use a different graph for inference__, 
e.g., use NHWC format, support encoded image format, etc. 
You can make these changes inside the `model` or `tower_func` in your `PredictConfig`.
The example in [examples/basics/export-model.py](../examples/basics/export-model.py) demonstrates such an altered inference graph.

### Exporter

In addition to the standard checkpoint format tensorpack saved for you during training,
you can also save your models into other formats so it may be more friendly for inference.

1. Export to `SavedModel` format for TensorFlow Serving:

   ```python
   from tensorpack.tfutils.export import ModelExporter
   ModelExporter(pred_config).export_serving('/path/to/export')
   ```

   This format contains both the graph and the variables. Refer to TensorFlow
   serving documentation on how to use it.

2. Export to a frozen and pruned graph for TensorFlow's builtin tools such as TOCO:

   ```python
   ModelExporter(pred_config).export_compact('/path/to/compact_graph.pb', toco_compatible=True)
   ```

   This format is just a serialized `tf.Graph`. The export process:
   - Converts all variables to constants to embed the variables directly in the graph.
   - Removes all unnecessary operations (training-only ops, e.g., learning-rate) to compress the graph.

   This creates a self-contained graph which includes all necessary information to run inference.
   
   To load the saved graph, you can simply:
   ```python
   graph_def = tf.GraphDef()
   graph_def.ParseFromString(open(graph_file, 'rb').read())
   tf.import_graph_def(graph_def)
   ```

[examples/basics/export-model.py](../examples/basics/export-model.py)
demonstrates the usage of such a frozen/pruned graph.
Again, you may often want to use a different graph for inference and you can
do so by the arguments of `PredictConfig`.

Note that the exporter relies on TensorFlow's automatic graph transformation, which do not always work reliably.
Automated graph transformation is often suboptimal or sometimes fail.
It's safer to write the graph by yourself.


## Inference After Training: Do It Yourself

Tensorpack is a training interface -- __it doesn't care what happened after training__.
During training it already provides everything you need for inference or model diagnosis after
training:

1. The model (the graph): you've already written it yourself with TF symbolic functions.
   Nothing about it is related to the tensorpack interface.
   If you use tensorpack layers, they are mainly just wrappers around `tf.layers`.

2. The trained parameters: tensorpack saves them in standard TF checkpoint format.
   Nothing about the format is related to tensorpack.

With the model and the trained parameters, you can do inference with whatever approaches
TensorFlow supports. Usually it involves the following steps:

### Step 1: build the model (graph)

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

    It's especially error-prone to load a metagraph on top of a non-empty graph.
    The potential name conflicts between the current graph and the nodes in the
    metagraph can lead to esoteric bugs or sometimes completely ruin the model.

	It's also very common to change the graph for inference.
	For example, you may need a different data layout for CPU inference,
	or you may need placeholders in the inference graph (which may not even exist in
	the training graph). However metagraph is not designed to be easily modified at all.

	Due to the above reasons, to do inference, it's best to recreate a clean graph (and save it if needed) by yourself.
```

### Step 2: load the checkpoint

You can just use `tf.train.Saver` for all the work.
Alternatively, use tensorpack's `SaverRestore(path).init(tf.get_default_session())`

Now, you've already built a graph for inference, and the checkpoint is loaded. 
You may now:

1. use `sess.run` to do inference
2. save the grpah to some formats for further processing
3. apply graph transformation for efficient inference

These steps are unrelated to tensorpack, and you'll need to learn TensorFlow and
do it yourself.
