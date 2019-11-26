
## Breaking API changes.

Tensorpack is in development, and API changes can happen.
The backward compatibilty will be __preserved for at least several months__, with a deprecation warning,
so you don't need to look at here very often.

Here are a list of things that were changed, starting from an early version.
TensorFlow itself also changes API and those are not listed here.

+ 2019/11/10. Drop Python 2 support.
+ [2019/03/20](https://github.com/tensorpack/tensorpack/commit/b8a50d72a7c655b6dc6facb17efd74069ba7f86c).
  The concept of `InputDesc` was replaced by its equivalent in TF:
  `tf.TensorSpec`. This may be a breaking change if you have customized
  code that relies on internals of `InputDesc`.
	To use `tf.TensorSpec` in your `ModelDesc`:
```python
    def inputs(self):
        return [tf.TensorSpec((None, 28, 28, 1), tf.float32, 'image'),
                tf.TensorSpec((None,), tf.int32, 'label')]
```
+ [2018/03/20] `ModelDesc` starts to use simplified interfaces:
	+ `_get_inputs()` renamed to `inputs()` and returns `tf.TensorSpec`.
	+ `build_graph(self, tensor1, tensor2)` returns the cost tensor directly.
	+ `_get_optimizer()` renamed to `optimizer()`.
	Old interface will still be available for a while, but new ones are recommended.
+ [2018/03/12] `JSONWriter` use a different file name, and will not automatically restore epoch number.
	`AutoResumeTrainConfig` was added to support resuming.
+ [2017/10/21]
	tensorpack is gradually switching to a new Trainer API.
	The old API will keep working for a while. See [issue](https://github.com/tensorpack/tensorpack/issues/458)
	for details.
+ [2017/10/18]
	`TrainConfig(predict_tower)` was deprecated. You can set the inference device directly when creating the `InferenceRunner` callback.
+ [2017/10/12](https://github.com/tensorpack/tensorpack/commit/7e963996f615b85f7459455596b4ee9bbd0bce8e).
	`tensorpack.RL` was deprecated. The RL examples are rewritten with OpenAI gym interface instead.
+ [2017/10/10](https://github.com/tensorpack/tensorpack/commit/7d40e049691d92018f50dc7d45bba5e8b140becc).
	`tfutils.distributions` was deprecated in favor of `tf.distributions` introduced in TF 1.3.
+ [2017/08/02](https://github.com/tensorpack/tensorpack/commit/875f4d7dbb5675f54eae5675fa3a0948309a8465).
	`Trainer.get_predictor` now takes GPU id. And `Trainer.get_predictors` was deprecated.
+ 2017/06/07. Now the library explicitly depends on msgpack-numpy>=0.3.9. The serialization protocol
	becomes incompatible if you've been using <0.3.9.
+ [2017/05/06](https://github.com/tensorpack/tensorpack/commit/0774ec66e66075486f6a36aba63cc2a151b9fec8).
	`replace_get_variable` was deprecated in favor of the official `custom_getter` interface.
	`{freeze,remap}_get_variable` was renamed to `{freeze,remap}_variables`.
+ [2017/04/09](https://github.com/tensorpack/tensorpack/commit/5beab907895aec36bdcaed62e25b976aad7979b8).
	`ParamRestore` was renamed to `DictRestore`.
+ [2017/03/16](https://github.com/tensorpack/tensorpack/commit/ccae46f4a3ca89dc3df901a338eef8447d19a730).
	`session_config` option in `PredictConfig` is deprecated. Use `session_creator` to define how to create session instead.
+ 2017/02/20. The interface of step callbacks are changed to be the same as `tf.train.SessionRunHook`.
	If you haven't written any custom step callbacks, there is nothing to do. Otherwise please refer
	to the [existing callbacks](https://github.com/tensorpack/tensorpack/blob/master/tensorpack/callbacks/steps.py).
+ [2017/02/12](https://github.com/tensorpack/tensorpack/commit/d1041a77a9c59d8c9abf64f389f3b605d65b483e).
	`TrainConfig(optimizer=)` was deprecated. Now optimizer is set in `ModelDesc`. And gradient processors become part of an optimizer.
* [2017/02/11](https://github.com/tensorpack/tensorpack/commit/5b29bda9f17d7b587259e13963c4c8093e8387f8).
	`_get_input_vars()` in `ModelDesc` was renamed to `_get_inputs`. `InputVar` was renamed to `InputDesc`.
* [2017/01/27](https://github.com/tensorpack/tensorpack/commit/a9dd0b8ec34209ab86a92875589dbbc4716e73ef).
	`TrainConfig(step_per_epoch)` was renamed to `steps_per_epoch`.
* [2017/01/25](https://github.com/tensorpack/tensorpack/commit/2df3dcf401a99fe61c699ad719e95528872d3abe).
	Argument order of `models.ConcatWith` is changed to follow the API change in TensorFlow upstream.
* [2017/01/25](https://github.com/tensorpack/tensorpack/commit/243e957fe6d62a0cfb5728bd77fb3e005d6603e4).
	`TrainConfig(callbacks=)` now takes a list of `Callback` instances.
* [2017/01/06](https://github.com/tensorpack/tensorpack/commit/bbf41d9e58053f843d0471e6d2d87ff714a79a90).
	`summary.add_moving_summary` now takes any number of positional arguments instead of a list.
* [2017/01/05](https://github.com/tensorpack/tensorpack/commit/651a5aea8f9aacad7147542021dcf106fc824bc2).
	The argument `TrainConfig(dataset=)` is renamed to `TrainConfig(dataflow=)`.
* [2016/12/15](https://github.com/tensorpack/tensorpack/commit/99c70935a7f72050f45891fbbcc49c4ce43aedce).
	The `predict_tower` option is in `TrainConfig` now instead of `Trainer`.
* [2016/11/10](https://github.com/tensorpack/tensorpack/commit/77bcc8b1afc984a569f6ec3eda0a3c47b4e2923a).
	The `{input,output}_var_names` argument in `PredictConfig` is renamed to `{input,output}_names`.
* [2016/11/06](https://github.com/tensorpack/tensorpack/commit/740e9d8ca146af5a911f68a369dd7348243a2253).
	The inferencer `ClassificationError` now expects the vector tensor returned by `prediction_incorrect` instead of the "wrong" tensor.
* [2016/10/17](https://github.com/tensorpack/tensorpack/commit/6eb0bebe60d6f38bcad9ddb3e6091b0b154a09cf).
	`Conv2D` and `FullyConnect` use `tf.identity` by default instead of `tf.nn.relu`.
* [2016/09/01](https://github.com/tensorpack/tensorpack/commit/fc9e45b0208ff09daf454d3bd910c540735b7f83).
	The method `_build_graph` of `ModelDesc` doesn't take `is_training` argument anymore.
	The `is_training` attribute can be obtained from tower context.
* [2016/05/15](https://github.com/tensorpack/tensorpack/commit/e69034b5c9b588db9fb52295b1e63c89e8b42654).
	The method `_get_cost` of `ModelDesc` is replaced by `_build_graph`.


