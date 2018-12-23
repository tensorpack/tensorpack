
ImageNet training code of ResNet, ShuffleNet, DoReFa-Net, AlexNet, Inception, VGG with tensorpack.

To train any of the models, just do `./{model}.py --data /path/to/ilsvrc`.
More options are available in `./{model}.py --help`.
Expected format of data directory is described in [docs](http://tensorpack.readthedocs.io/modules/dataflow.dataset.html#tensorpack.dataflow.dataset.ILSVRC12).
Some pretrained models can be downloaded at [tensorpack model zoo](http://models.tensorpack.com/).

### ShuffleNet

Reproduce ImageNet results of the following two papers:
+ [ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices](https://arxiv.org/abs/1707.01083)
+ [ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design](https://arxiv.org/abs/1807.11164)

| Model                                                                                                    | Flops | Top-1 Error | Paper's Error | Flags         |
|:---------------------------------------------------------------------------------------------------------|:------|:-----------:|:-------------:|:-------------:|
| ShuffleNetV1 0.5x  [:arrow_down:](http://models.tensorpack.com/ImageNetModels/ShuffleNetV1-0.5x-g=8.npz) | 40M   | 40.8%       | 42.3%         | `-r=0.5`      |
| ShuffleNetV1 1x    [:arrow_down:](http://models.tensorpack.com/ImageNetModels/ShuffleNetV1-1x-g=8.npz)   | 140M  | 32.6%       | 32.4%         | `-r=1`        |
| ShuffleNetV2 0.5x  [:arrow_down:](http://models.tensorpack.com/ImageNetModels/ShuffleNetV2-0.5x.npz)     | 41M   | 39.5%       | 39.7%         | `-r=0.5 --v2` |
| ShuffleNetV2 1x    [:arrow_down:](http://models.tensorpack.com/ImageNetModels/ShuffleNetV2-1x.npz)       | 146M  | 30.6%       | 30.6%         | `-r=1 --v2`   |

To print flops:
```bash
./shufflenet.py --flops [--other-flags]
```

Download and evaluate a pretrained model:
```
wget http://models.tensorpack.com/ImageNetModels/ShuffleNetV2-0.5x.npz
./shufflenet.py --eval --data /path/to/ilsvrc --load ShuffleNetV2-0.5x.npz --v2 -r=0.5
```

### AlexNet

This AlexNet script is quite close to the settings in its [original
paper](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks).
Trained with 2 GPUs and 64 batch size per GPU, the script reaches 58% single-crop validation
accuracy after 100 epochs (21h on 2 V100s).
It also puts in tensorboard the first-layer filter visualizations similar to the paper.
See `./alexnet.py --help` for usage.

### VGG16

This VGG16 script, when trained with 8 GPUs and 32 batch size per GPU, reaches the following
validation error after 100 epochs (30h with 8 P100s). This reproduces the VGG
experiments in the paper [Group Normalization](https://arxiv.org/abs/1803.08494).
See `./vgg16.py --help` for usage.

 | No Normalization                          | Batch Normalization | Group Normalization |
 |:------------------------------------------|:-------------------:|:-------------------:|
 | 29~30% (large variation with random seed) | 28%                 | 27.6%               |

Note that the purpose of this experiment in the paper is not to claim GroupNorm
has better performance than BatchNorm.

### Inception-BN

This Inception-BN script reaches 27% single-crop validation error after 300k steps with 6 GPUs.
The training recipe is very different from the original paper because the paper
is a bit vague on these details.

### ResNet

See [ResNet examples](../ResNet). It includes variants like pre-activation
ResNet, squeeze-and-excitation networks.

### DoReFa-Net

See [DoReFa-Net examples](../DoReFa-Net).
It includes other quantization methods such as Binary Weight Network, Trained Ternary Quantization.

