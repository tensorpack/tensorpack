
ImageNet training code of ResNet, ShuffleNet, DoReFa-Net, AlexNet, Inception, VGG with tensorpack.

To train any of the models, just do `./{model}.py --data /path/to/ilsvrc`.
Expected format of data directory is described in [docs](http://tensorpack.readthedocs.io/en/latest/modules/dataflow.dataset.html#tensorpack.dataflow.dataset.ILSVRC12).
Pretrained models can be downloaded at [tensorpack model zoo](http://models.tensorpack.com/).

### ShuffleNet

Reproduce [ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices](https://arxiv.org/abs/1707.01083)
on ImageNet.

This is a 38Mflops ShuffleNet, corresponding to `ShuffleNet 0.5x g=3` in __the
2nd arxiv version__ of the paper.
After 240 epochs (36 hours on 8 P100s) it reaches top-1 error of 42.32%,
matching the paper's number.

To print flops:
```bash
./shufflenet.py --flops
```
It will print about 75Mflops, because the paper counts multiply+add as 1 flop.

Evaluate the [pretrained model](http://models.tensorpack.com/ShuffleNet/):
```
./shufflenet.py --eval --data /path/to/ilsvrc --load /path/to/model
```

### AlexNet

This AlexNet script is quite close to the setting in its [original
paper](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks).
Trained with 64x2 batch size, the script reaches 58% single-crop validation
accuracy after 100 epochs. It also generates first-layer filter visualizations
similar to the paper in tensorboard.

### Inception-BN, VGG16

This Inception-BN script reaches 27% single-crop validation error after 300k steps with 6 GPUs.
The training recipe is very different from the original paper because the paper
is a bit vague on these details.

This VGG16 script, when trained with 32x8 batch size, reaches the following
validation error after 100 epochs (30h with 8 P100s). This is the code for the VGG
experiments in the paper [Group Normalization](https://arxiv.org/abs/1803.08494).

 | No Normalization                          | Batch Normalization | Group Normalization |
 |:------------------------------------------|---------------------|--------------------:|
 | 29~30% (large variation with random seed) | 28%                 |               27.6% |

### ResNet

See [ResNet examples](../ResNet). It includes variants like pre-activation
ResNet, squeeze-and-excitation networks.

### DoReFa-Net

See [DoReFa-Net examples](../DoReFa-Net).
It includes other quantization methods such as Binary Weight Network, Trained Ternary Quantization. 

