
## imagenet-resnet.py

__Training__ code of three variants of ResNet on ImageNet:

* [Original ResNet](https://arxiv.org/abs/1512.03385)
* [Pre-activation ResNet](https://arxiv.org/abs/1603.05027)
* [Squeeze-and-Excitation ResNet](https://arxiv.org/abs/1709.01507)

The training mostly follows the setup in [fb.resnet.torch](https://github.com/facebook/fb.resnet.torch)
and gets similar performance (with much fewer lines of code).
Models can be [downloaded here](https://goo.gl/6XjK9V).

| Model              | Top 5 Error | Top 1 Error |
|:-------------------|-------------|------------:|
| ResNet18           |     10.50%  |      29.66% |
| ResNet34					 |     8.56%   |      26.17% |
| ResNet50           |     6.85%   |      23.61% |
| ResNet50-SE				 |     6.24%   |      22.64% |
| ResNet101      		 |     6.04%   |      21.95% |
| ResNet152      		 |     5.78%   |      21.51% |

To train, just run:
```bash
./imagenet-resnet.py --data /path/to/original/ILSVRC --gpu 0,1,2,3 -d 50 [--mode resnet/preact/se]
```
You should be able to see good GPU utilization (95%~99%), if your data is fast enough.
The default data pipeline is probably OK for most systems.
See the [tutorial](http://tensorpack.readthedocs.io/en/latest/tutorial/efficient-dataflow.html) on other options to speed up your data.

![imagenet](imagenet-resnet.png)

## load-resnet.py

This script only converts and runs ImageNet-ResNet{50,101,152} Caffe models [released by MSRA](https://github.com/KaimingHe/deep-residual-networks).
Note that the architecture is different from the `imagenet-resnet.py` script and the models are not compatible.

Usage:
```bash
# download and convert caffe model to npy format
python -m tensorpack.utils.loadcaffe PATH/TO/{ResNet-101-deploy.prototxt,ResNet-101-model.caffemodel} ResNet101.npy
# run on an image
./load-resnet.py --load ResNet-101.npy --input cat.jpg --depth 101
```

The converted models are verified on ILSVRC12 validation set.
The per-pixel mean used here is slightly different from the original.

| Model              | Top 5 Error | Top 1 Error |
|:-------------------|-------------|------------:|
| ResNet 50          |      7.78%  |      24.77% |
| ResNet 101         |      7.11%  |      23.54% |
| ResNet 152         |      6.71%  |      23.21% |

## cifar10-resnet.py

Reproduce pre-activation ResNet on CIFAR10.

![cifar10](cifar10-resnet.png)

Also see a [DenseNet implementation](https://github.com/YixuanLi/densenet-tensorflow) of the paper [Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993).


## cifar10-preact18-mixup.py

Reproduce mixup pre-activation ResNet18 on CIFAR10.
Please notice that this preact18 architecture is
[different](https://github.com/kuangliu/pytorch-cifar/blob/master/models/preact_resnet.py)
as the [mixup paper](https://arxiv.org/abs/1710.09412) said.

Usage:
```bash
./cifar10-preact18-mixup.py  # train without mixup
./cifar10-preact18-mixup.py --mixup	 # with mixup
```

Validation error with the original LR schedule (100-150-200): __5.0%__ without mixup, __3.8%__ with mixup.

With 2x LR schedule: 4.7% without mixup, and 3.2% with mixup.
