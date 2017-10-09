
## ShuffleNet

Reproduce [ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices](https://arxiv.org/abs/1707.01083)
on ImageNet.

This is a 40Mflops ShuffleNet,
roughly corresponding to `ShuffleNet 0.5x (arch2)	g=8` in the paper.
But detailed architecture may not be the same.
After 100 epochs it reaches top-1 error of 42.62.

### Usage:

Print flops with tensorflow:
```bash
./shufflenet.py --flops
```
It will print about 80Mflops, because TF counts FMA as 2 flops while the paper counts it as 1 flop.

Train (takes 24 hours on 8 Maxwell TitanX):
```bash
./shufflenet.py --data /path/to/ilsvrc/
```
