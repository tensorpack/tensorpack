

## Dynamic Filter Networks

Reproduce the "Learning steering filters" experiments in
[Dynamic Filter Networks](https://arxiv.org/abs/1605.09673).

The input image is convolved by a dynamically learned filter to match
a ground truth image.
In the end the filters converge to the true steering filter used to generate the ground truth.

This also demonstrates how to put images into tensorboard directly from Python.
![filters](https://cloud.githubusercontent.com/assets/6756603/26296499/384845ca-3ecf-11e7-8fa5-df2e322b3a3d.gif)

To run:
```bash
./steering-filter.py --gpu 0
```
