
# Trainers

Tensorpack trainers contain logic of:

1. Building the graph.
2. Running the iterations (with callbacks).

Usually you won't touch these methods directly, but use
[higher-level interface](training-interface.html) on trainers.
You'll only need to __select__ what trainer to use.
But some basic knowledge of how they work is useful:

### Tower Trainer

[TowerTrainer](../modules/train.html#tensorpack.train.TowerTrainer)
is a trainer that uses "tower function" to build models.
All existing trainers in tensorpack are subclass of ``TowerTrainer``,
because this concept is able to cover most types of neural-network training tasks.

#### What is Tower Function

Following the terminology in TensorFlow,
a __tower function__ is a callable that takes input tensors and adds __one replicate__ of the model to the graph.

The concept of tower is used mainly to support:
1. Data-parallel multi-GPU training, where a replicate is built on each GPU.
2. Graph construction for inference, where a replicate is built under inference mode.

A user needs to provide a tower function to use `TowerTrainer`.
In particular, when working with the `ModelDesc` interface, the `build_graph`
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
3. It must __respect variable scopes__:
   * The name of any trainable variables created in the function must be like "variable_scope_name/custom/scopes/name".
     Don't depend on name_scope's name. Don't depend on some tensor's name. Don't use variable_scope's name twice.
   * The creation of any trainable variables must __respect reuse__ variable scope.
     To respect variable reuse (i.e. sharing), use `tf.get_variable` instead of `tf.Variable` in the function.

     On the other hand, for a non-trainable variable, it may be desirable to not reuse it between towers.
     In this case, `tf.Variable` can be used to ensure creation of new variables in each tower even when `reuse=True`.
   * Do not modify the reuse option (e.g., by `scope.reuse_variables()`) of a variable
     scope that is not created by you. This affects other's code. You can always
     open new scopes if you need the reuse option.
4. It cannot create scopes or variables containing the name 'tower', as it is
   reserved for special use.
     
These conventions are easy to follow, and most layer wrappers (e.g.,
tf.layers/slim/tensorlayer) do follow them. Note that certain Keras layers do not
follow these conventions and will need some workarounds if used within tensorpack.

#### What You Can Do Inside Tower Function
1. Call any symbolic functions as long as they follow the above rules.
2. The tower function will be called under a
 [TowerContext](../modules/tfutils.html#tensorpack.tfutils.tower.BaseTowerContext),
 which can be accessed by [get_current_tower_context()](../modules/tfutils.html#tensorpack.tfutils.tower.get_current_tower_context).
   The context contains information about training/inference mode, scope name, etc.
   You can use the context to build a different graph under different mode.


### Multi-GPU Trainers

For data-parallel multi-GPU training, different [multi-GPU trainers](../modules/train.html)
implement different distribution strategies.
They take care of device placement, gradient averaging and synchronoization
in the efficient way and all reach the same performance as the
[official TF benchmarks](https://www.tensorflow.org/performance/benchmarks).
It takes only one line of code change to use them, e.g. `trainer=SyncMultiGPUTrainerReplicated(...)`.

Note some __common problems__ when using these trainers:

1. In each iteration, instead of taking one input tensor for all GPUs and split,
    all GPUs take tensors from the `InputSource`.
	So the total batch size across all GPUs would become ``(batch size of InputSource) * #GPU``.

    ```eval_rst
    .. note:: 

        Splitting a tensor for data-parallel training (as done by frameworks like Keras) 
        makes no sense at all. 
        First, it wastes time doing the split because typically data is first concatenated by the user.
        Second, this puts unnecessary shape constraints on the data, that the
        inputs on each GPU needs to have compatible shapes.
    ```

2. The tower function (your model code) will get called once on each GPU.
   You must follow the abovementieond rules of tower function.

### Distributed Trainers

Distributed training needs the [horovod](https://github.com/uber/horovod) library which offers high-performance allreduce implementation.
To run distributed training, first install horovod properly, then refer to the
documentation of [HorovodTrainer](../modules/train.html#tensorpack.train.HorovodTrainer).

Tensorpack has implemented some other distributed trainers using TF's native API,
but TensorFlow is not actively supporting its distributed training features, and
its native distributed performance isn't very good even today.
Therefore those trainers are not actively maintained and are __not recommended for use__.
