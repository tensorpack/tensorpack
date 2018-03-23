
ImageNet training code of ResNet, Inception, VGG, ShuffleNet, DoReFa-Net with tensorpack.

To train any of the models, just do `./{model}.py --data /path/to/ilsvrc`.
Expected format of data directory is described in [docs](http://tensorpack.readthedocs.io/en/latest/modules/dataflow.dataset.html#tensorpack.dataflow.dataset.ILSVRC12).
Pretrained models can be downloaded at [tensorpack model zoo](http://models.tensorpack.com/).

### ShuffleNet

Reproduce [ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices](https://arxiv.org/abs/1707.01083)
on ImageNet.

This is a 38Mflops ShuffleNet, corresponding to `ShuffleNet 0.5x g=3` in [version 2](https://arxiv.org/pdf/1707.01083v2) of the paper.
After 240 epochs (36 hours on 8 P100s) it reaches top-1 error of 42.32%, better than the paper's number.

To print flops:
```bash
./shufflenet.py --flops
```
It will print about 75Mflops, because the paper counts multiply+add as 1 flop.

Evaluate the [pretrained model](http://models.tensorpack.com/ShuffleNet/):
```
./shufflenet.py --eval --data /path/to/ilsvrc --load /path/to/model
```

### Inception-BN, VGG16

This Inception-BN script reaches 27% single-crop error after 300k steps with 6 GPUs.

This VGG16 script, when trained with 32x8 batch size, reaches 29~30% single-crop error after 100 epochs (30h with 8 P100s),
28% with BN, and 27.6% with GN.

### ResNet, DoReFa-Net

See [ResNet examples](../ResNet) and [DoReFa-Net examples](../DoReFa-Net).
