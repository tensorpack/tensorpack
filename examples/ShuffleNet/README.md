
## ShuffleNet

Reproduce [ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices](https://arxiv.org/abs/1707.01083)
on ImageNet.

This is a 38Mflops ShuffleNet, corresponding to `ShuffleNet 0.5x g=3` in [version 2](https://arxiv.org/pdf/1707.01083v2) of the paper.
After 240 epochs it reaches top-1 error of 42.32, better than the paper's number.

### Usage:

Print flops with tensorflow:
```bash
./shufflenet.py --flops
```
It will print about 75Mflops, because the paper counts multiply+add as 1 flop.

Train (takes 36 hours on 8 P100s):
```bash
./shufflenet.py --data /path/to/ilsvrc/
```

Evaluate the [pretrained model](http://models.tensorpack.com/ShuffleNet/):
```
./shufflenet.py --eval --data /path/to/ilsvrc --load /path/to/model
```
