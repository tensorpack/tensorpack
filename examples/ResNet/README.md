
ImageNet training code coming soon.

## cifar10-resnet.py

Reproduce the results in paper "Deep Residual Learning for Image Recognition", [http://arxiv.org/abs/1512.03385](http://arxiv.org/abs/1512.03385)
with the variants proposed in "Identity Mappings in Deep Residual Networks", [https://arxiv.org/abs/1603.05027](https://arxiv.org/abs/1603.05027) on CIFAR10.

The train error shown here is a moving average of the error rate of each batch in training.
The validation error here is computed on test set.

![cifar10](cifar10-resnet.png)

Download model:
[Cifar10 ResNet-110 (n=18)](https://drive.google.com/open?id=0B9IPQTvr2BBkTXBlZmh1cmlnQ0k)

Also see an implementation of [DenseNet](https://github.com/YixuanLi/densenet-tensorflow) from [Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993).

## load-resnet.py

A script to convert and run ResNet{50,101,152} ImageNet models released by Kaiming.

Example usage:
```bash
python -m tensorpack.utils.loadcaffe PATH/TO/{ResNet-101-deploy.prototxt,ResNet-101-model.caffemodel} ResNet101.npy
./load-resnet.py --load ResNet-101.npy --input cat.png --depth 101
```

