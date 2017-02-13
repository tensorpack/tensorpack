
## Breaking API changes.

tensorpack is still in early development, and API changes can happen.
The backward compatibilty will be __preserved for several months__, with a deprecation warning,
so you won't need to look at here very often.

Here are a list of things that were changed, starting from an early version.
TensorFlow itself also changes API and those are not listed here.

+ 2017/02/12. `TrainConfig(optimizer=)` was deprecated. Now optimizer is set in `ModelDesc`. And
	gradient processors become part of an optimizer. See [commit](https://github.com/ppwwyyxx/tensorpack/commit/d1041a77a9c59d8c9abf64f389f3b605d65b483e).
* 2017/02/11. `_get_input_vars()` in `ModelDesc` was renamed to `_get_inputs`. `InputVar` was
	renamed to `InputDesc`. See [commit](https://github.com/ppwwyyxx/tensorpack/commit/5b29bda9f17d7b587259e13963c4c8093e8387f8).
* 2017/01/27. `TrainConfig(step_per_epoch)` was renamed to `steps_per_epoch`. See [commit](https://github.com/ppwwyyxx/tensorpack/commit/a9dd0b8ec34209ab86a92875589dbbc4716e73ef).
* 2017/01/25. Argument order of `models.ConcatWith` is changed to follow the API change in
	TensorFlow upstream. See [commit](https://github.com/ppwwyyxx/tensorpack/commit/2df3dcf401a99fe61c699ad719e95528872d3abe).
* 2017/01/25. `TrainConfig(callbacks=)` now takes a list of `Callback` instances. See [commit](https://github.com/ppwwyyxx/tensorpack/commit/243e957fe6d62a0cfb5728bd77fb3e005d6603e4)
	on how to change your code.
* 2017/01/06. `summary.add_moving_summary` now takes any number of positional arguments instead of a list.
	See [commit](https://github.com/ppwwyyxx/tensorpack/commit/bbf41d9e58053f843d0471e6d2d87ff714a79a90) to change your code.
* 2017/01/05. The argument `TrainConfig(dataset=)` is renamed to `TrainConfig(dataflow=)`.
	See [commit](https://github.com/ppwwyyxx/tensorpack/commit/651a5aea8f9aacad7147542021dcf106fc824bc2) to change your code.
* 2016/12/15. The `predict_tower` option is in `TrainConfig` now instead of `Trainer`. See
	[commit](https://github.com/ppwwyyxx/tensorpack/commit/99c70935a7f72050f45891fbbcc49c4ce43aedce).
* 2016/11/10. The `{input,output}_var_names` argument in `PredictConfig` is renamed to `{input,output}_names`. See [commit](https://github.com/ppwwyyxx/tensorpack/commit/77bcc8b1afc984a569f6ec3eda0a3c47b4e2923a).
* 2016/11/06. The inferencer `ClassificationError` now expects the vector tensor returned by
	`prediction_incorrect` instead of the "wrong" tensor. See [commit](https://github.com/ppwwyyxx/tensorpack/commit/740e9d8ca146af5a911f68a369dd7348243a2253)
	to make changes.
* 2016/10/17. `Conv2D` and `FullyConnect` use `tf.identity` by default instead of `tf.nn.relu`.
	See [commit](https://github.com/ppwwyyxx/tensorpack/commit/6eb0bebe60d6f38bcad9ddb3e6091b0b154a09cf).
* 2016/09/01. The method `_build_graph` of `ModelDesc` doesn't take `is_training` argument anymore.
	The `is_training` attribute can be obtained from tower context. See [commit](https://github.com/ppwwyyxx/tensorpack/commit/fc9e45b0208ff09daf454d3bd910c540735b7f83).
* 2016/05/15. The method `_get_cost` of `ModelDesc` is replaced by `_build_graph`. See [commit](https://github.com/ppwwyyxx/tensorpack/commit/e69034b5c9b588db9fb52295b1e63c89e8b42654).


