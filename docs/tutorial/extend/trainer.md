## Understand Trainer

### How Existing (Single-Cost) Trainers Work

Most neural network training tasks are single-cost optimization.
Tensorpack provides some trainer implementations for such tasks.
These trainers will take care help you define the graph, with the following arguments:

1. Some `tf.TensorSpec`, the signature of the input.
2. An `InputSource`, where the input come from. See [Input Pipeline](./input-source.md).
3. A function which takes input tensors and returns the cost.
4. A function which returns an optimizer.

These are documented in [SingleCostTrainer.setup_graph](/modules/train.html#tensorpack.train.SingleCostTrainer.setup_graph).
In practice you'll not use this method directly, but use [high-level interface](/tutorial/training-interface.html#with-modeldesc-and-trainconfig) instead.


### Tower Trainer

[TowerTrainer](../modules/train.html#tensorpack.train.TowerTrainer)
is a trainer that uses user-provided "tower function" to build models.
All existing trainers in tensorpack are subclass of ``TowerTrainer``,
because this concept is able to cover most types of neural-network training tasks.

#### What is Tower Function

Following the terminology in TensorFlow,
a __tower function__ is a callable that takes input tensors and adds __one replicate__ of the model to the graph.
In short, __tower function builds your model__.
If you can write a function that builds your model, then you can use `TowerTrainer`.

The concept of "tower" is used mainly to support:
1. Data-parallel multi-GPU training, where a replicate is built on each GPU.
2. Graph construction for inference, where a replicate is built under inference mode.

A user needs to provide a tower function to use `TowerTrainer`.
In particular, when working with the commonly used `ModelDesc` interface, the `build_graph`
method will be part of the tower function.

#### Rules of Tower Function

The tower function needs to follow some rules:

1. __It may get called multiple times__ for data-parallel training or inference. As a result:
   * You'll need to be careful when modifying global states, e.g.
     adding ops to collections, setting attributes of a model instance.
   * To use a tensorflow-hub module, you need to initialize the
     module outside the tower function, and call the module inside the tower function.
2. It must __respect variable collections__:
   * (Required) Only put variables __trainable by gradient descent__ into `TRAINABLE_VARIABLES`.
   * (Recommended) Put non-trainable variables that need to be used in inference into `MODEL_VARIABLES`.
3. It must __respect variable scope names__:

   The name of any trainable variables created in the function must be like "variable_scope_name/other/scopes/and/name".
	 Strictly speaking, the name of any trainable variables must:

     * Start with the name of the enclosing variable_scope when the tower function is called.
	 * Not use the same variable_scope's name twice in its name.
	 * Not depend on name_scope's name.
	 * Not depend on any tensor's name (because the tensor's name may depend on name_scope's name).

	 Tensorpack layers create variables based on the name given to the layer:
	 e.g., `Conv2D('test', x)` will open a variable scope named "test".
     In order to respect the above rules,
	 the name of the layer must not depend on name_scope's name or any tensor's name.
4. It must __respect variable scope reuse__:
   * The creation of any trainable variables must __respect reuse__ variable scope.
     To respect variable reuse (i.e. sharing), use `tf.get_variable` instead of `tf.Variable` in the function.

     On the other hand, for a non-trainable variable, it may be desirable to not reuse it between towers.
     In this case, `tf.Variable` can be used to ensure creation of new variables in each tower even when `reuse=True`.
   * Do not modify the reuse option (e.g., by `scope.reuse_variables()`) of a variable
     scope that is not created by you. This affects other's code. You can always
     open new scopes if you need the reuse option.
5. It must not create scopes or variables containing the name 'tower', as it is
   reserved for special use.

These conventions are easy to follow, and most layer wrappers (e.g.,
tf.layers/slim/tensorlayer) do follow them. Note that certain Keras layers do not
follow these conventions and will need some workarounds if used within tensorpack.

#### What You Can Do Inside a Tower Function
1. Call any symbolic functions as long as they follow the above rules.
2. The tower function will be called under a
 [TowerContext](../modules/tfutils.html#tensorpack.tfutils.tower.BaseTowerContext),
 which can be accessed by [get_current_tower_context()](../modules/tfutils.html#tensorpack.tfutils.tower.get_current_tower_context).
   The context contains information about training/inference mode, scope name, etc.
   You can use the context to build a different graph under different mode.


### Write a Trainer

The existing trainers should be enough for data-parallel single-cost optimization tasks.
If you just want to do some extra work during training, first consider writing it as a callback,
or write an issue to see if there is a better solution than creating new trainers.
If your task is fundamentally different from single-cost optimization, you will need to write a trainer.

You can customize the trainer by either using or inheriting the `Trainer`/`TowerTrainer` class.
You will need to do two things for a new Trainer:

1. Define the graph. There are 2 ways you can do this:
    1. Create any tensors and ops you need, before creating the trainer.
    2. Create them inside `Trainer.__init__`.

2. Define what is the iteration. There are 2 ways to define the iteration:
	1. Set `Trainer.train_op` to a TensorFlow operation. This op will be run by default.
	2. Subclass `Trainer` and override the `run_step()` method. This way you can
       do something more than running an op.

       Note that trainer has `self.sess` and `self.hooked_sess`: only the hooked
       session will trigger the `before_run`/`after_run` callbacks.
       If you need more than one `Session.run` in one steps, special care needs
       to be taken to choose which session to use, because many states
       (global steps, StagingArea, summaries) are maintained through `before_run`/`after_run`.

If you want to write a new trainer,
Tensorpack examples include several different
[GAN trainers](../../examples/GAN/GAN.py) for a reference.
