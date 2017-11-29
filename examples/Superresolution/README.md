## Superresolution

Reproduce the "Efficient Sub-Pixel Convolutional Neural Network" experiments in
<https://arxiv.org/abs/1609.05158>
by Wenzhe Shi, et al.

Given an low-resolution image, the network is trained to
produce an high resolution image using a pixel-shift layer.

<p align="center"> <img src="./demo.jpg" width="100%"> </p>

* Left: input image (upscaled with bi-cubic interpolation).
* Middle: prediction of the network.
* Right: ground-truth image.

This further illustrates training on MS COCO and running inference on arbitrary image files.

To train download MS COCO dataset

```bash
DBDIR=/datasets/mscoco/
cd ${DBDIR}
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
python data_sampler.py --lmdb val2017.lmdb --input val2017.zip --create
python data_sampler.py --lmdb train2017.lmdb --input train2017.zip --create
```

and then train the model using

```bash
python realtime_superresolution.py --gpu 0 -- lmdb_path ${DBDIR}
```

Inference can be done by

```bash
python realtime_superresolution.py --apply \
   --load /train_log/realtime_superresolution/checkpoint \
   --highres monarch.bmp --output monarch --gpu 0
```

which gives 3 files

- monarchbaseline.png
- monarchgroundtruth.png
- monarchprediction.png