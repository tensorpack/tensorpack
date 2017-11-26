
## Holistically-Nested Edge Detection

Reproduce the HED paper by Saining. See [https://arxiv.org/abs/1504.06375](https://arxiv.org/abs/1504.06375).

![HED](demo.jpg)

(Bottom-left: raw fused heatmap; Middle and right column: raw heatmaps at different stages)

HED is a fully-convolutional architecture. This code generally would also work
for other FCN tasks such as semantic segmentation and detection.

## Usage

This script only needs the original BSDS dataset and applies augmentation on the fly.
It will automatically download the dataset to `$TENSORPACK_DATASET/` if not there.

It requires pretrained vgg16 model. See the docs in [examples/load-vgg16.py](../load-vgg16.py)
for instructions to convert from vgg16 caffe model.

To view augmented training images:
```bash
./hed.py --view
```

To start training:
```bash
./hed.py --load vgg16.npy
```
It takes about 100k steps (~10 hours on a TitanX) to reach a reasonable performance.

To inference (produce a heatmap at each level at out*.png):
```bash
./hed.py --load pretrained.model --run a.jpg
```
Models I trained can be downloaded [here](http://models.tensorpack.com/HED/).

To view the loss curve:
```bash
cat train_log/hed/stat.json | jq '.[] |
"\(.xentropy1)\t\(.xentropy2)\t\(.xentropy3)\t\(.xentropy4)\t\(.xentropy5)\t\(.xentropy6)"' -r | \
				tpk-plot-point --legend 1,2,3,4,5,final --decay 0.8
```
Or just open tensorboard.
