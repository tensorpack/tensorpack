
## Keras + Tensorpack

Use Keras to define a model a train it with efficient tensorpack trainers.


### Simple Examples:

[mnist-keras.py](mnist-keras.py): a simple MNIST model written mostly in tensorpack style, but use Keras model as symbolic functions.
[mnist-keras-v2.py](mnist-keras-v2.py): the same MNIST model written in Keras style.

### ImageNet Example:

[imagenet-resnet-keras.py](imagenet-resnet-keras.py):
reproduce exactly the same setting of [tensorpack ResNet example](../ResNet) on ImageNet.
It has:

+ ResNet-50 model modified from [keras.applications](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/keras/_impl/keras/applications/resnet50.py)
+ Multi-GPU data-parallel __training and validation__ which scales (With 8 V100s, still has >90% GPU utilization and finished training in 21 hours)
+ Good accuracy (same as the [tensorpack ResNet example](../ResNet))
