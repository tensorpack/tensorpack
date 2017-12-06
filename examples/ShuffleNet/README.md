
## ShuffleNet

Reproduce [ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices](https://arxiv.org/abs/1707.01083)
on ImageNet.

This is a 40Mflops ShuffleNet, corresponding to `ShuffleNet 0.5x (arch2)	g=8` in the paper.
After 100 epochs it reaches top-1 error of 42.62, matching the paper's number.

### Usage:

Print flops with tensorflow:
```bash
./shufflenet.py --flops
```
It will print about 80Mflops, because the paper counts multiply+add as 1 flop.

Train (takes 24 hours on 8 Maxwell TitanX):
```bash
./shufflenet.py --data /path/to/ilsvrc/
```

Eval the [pretrained model](http://models.tensorpack.com/ShuffleNet/):
```
./shufflenet.py --eval --data /path/to/ilsvrc --load /path/to/model
```
