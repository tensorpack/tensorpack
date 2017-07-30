
## imagenet-resnet.py, imagenet-resnet-se.py

__Training__ code of ResNet on ImageNet, with pre-activation and squeeze-and-excitation.
The pre-act ResNet follows the setup in [fb.resnet.torch](https://github.com/facebook/fb.resnet.torch) (except for the weight decay)
and gets similar performance (with much fewer lines of code).
Models can be [downloaded here](https://goo.gl/6XjK9V).

| Model              | Top 5 Error | Top 1 Error |
|:-------------------|-------------|------------:|
| ResNet 18          |      10.67% |      29.50% |
| ResNet 34          |      8.66%  |      26.45% |
| ResNet 50          |      7.13%  |      24.12% |
| ResNet 101         |      6.54%  |      22.89% |

To train, just run:
```bash
./imagenet-resnet.py --data /path/to/original/ILSVRC --gpu 0,1,2,3 -d 18
```
You should be able to see good GPU utilization (around 95%), if your data is fast enough.
See the [tutorial](http://tensorpack.readthedocs.io/en/latest/tutorial/efficient-dataflow.html) on how to speed up your data.

![imagenet](imagenet-resnet.png)

## load-resnet.py

This script only converts and runs ImageNet-ResNet{50,101,152} Caffe models [released by Kaiming](https://github.com/KaimingHe/deep-residual-networks).

Example usage:
```bash
# convert caffe model to npy format
python -m tensorpack.utils.loadcaffe PATH/TO/{ResNet-101-deploy.prototxt,ResNet-101-model.caffemodel} ResNet101.npy
# run on an image
./load-resnet.py --load ResNet-101.npy --input cat.jpg --depth 101
```

The converted models are verified on ILSVRC12 validation set.
The per-pixel mean used here is slightly different from the original.

| Model              | Top 5 Error | Top 1 Error |
|:-------------------|-------------|------------:|
| ResNet 50          |      7.89%  |      25.03% |
| ResNet 101         |      7.16%  |      23.74% |
| ResNet 152         |      6.81%  |      23.28% |

## cifar10-resnet.py

Reproduce pre-activation ResNet on CIFAR10.

![cifar10](cifar10-resnet.png)

Also see a [DenseNet implementation](https://github.com/YixuanLi/densenet-tensorflow) of the paper [Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993).
