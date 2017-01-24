
## Breaking API changes.

tensorpack is still in early development, and API changes can happen.
Usually the backward compatibilty is preserved for several month, with a deprecation warning.
If you are an early bird to try out this library, you might need to occasionally update your code.

Here are a list of things that were changed, starting from an early version.
TensorFlow itself also changes API and those are not listed here.


* 2017/01/06. `summary.add_moving_summary` now takes any number of positional arguments instead of a list.
	See [commit](https://github.com/ppwwyyxx/tensorpack/commit/bbf41d9e58053f843d0471e6d2d87ff714a79a90) to change your code.
* 2017/01/05. The argument `TrainConfig(dataset=)` is renamed to `TrainConfig(dataflow=)`.
	See [commit](https://github.com/ppwwyyxx/tensorpack/commit/651a5aea8f9aacad7147542021dcf106fc824bc2) to change your code.
* 2016/11/06. The inferencer `ClassificationError` now expects the vector tensor returned by
	`prediction_incorrect` instead of the "wrong" tensor. See [commit](https://github.com/ppwwyyxx/tensorpack/commit/740e9d8ca146af5a911f68a369dd7348243a2253)
	to make changes.
* 2016/10/17. `Conv2D` and `FullyConnect` use `tf.identity` by default instead of `tf.nn.relu`.
	See [commit](https://github.com/ppwwyyxx/tensorpack/commit/6eb0bebe60d6f38bcad9ddb3e6091b0b154a09cf).
* 2016/09/01. The method `_build_graph` of `ModelDesc` doesn't takes `is_training` argument anymore.
	The `is_training` attribute can be obtained from tower context. See [commit](https://github.com/ppwwyyxx/tensorpack/commit/fc9e45b0208ff09daf454d3bd910c540735b7f83).
* 2016/05/15. The method `_get_cost` of `ModelDesc` is replaced by `_build_graph`. See [commit](https://github.com/ppwwyyxx/tensorpack/commit/e69034b5c9b588db9fb52295b1e63c89e8b42654).


