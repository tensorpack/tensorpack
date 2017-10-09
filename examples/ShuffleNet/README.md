
## ShuffleNet

Reproduce [ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices](https://arxiv.org/abs/1707.01083)
on ImageNet.

This is a 40MFlops ShuffleNet,
roughly corresponding to `ShuffleNet 0.5x (arch2)	g=8` in the paper.
But detailed architecture may not be the same.
After 100 epochs it reaches top-1 error of 42.62.

### Usage:

Print flops with tensorflow:
```bash
./shufflenet.py --flops
```
It will print about 80MFlops, because TF counts FMA as 2 flops while the paper counts it as 1 flop.

Train:
```bash
./shufflenet.py --data /path/to/ilsvrc/
```
