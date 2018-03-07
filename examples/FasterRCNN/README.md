# Faster-RCNN / Mask-RCNN on COCO
This example aims to provide a minimal (1.3k lines) implementation of
end-to-end Faster-RCNN & Mask-RCNN (with ResNet backbones) on COCO.

## Dependencies
+ Python 3; TensorFlow >= 1.4.0
+ [pycocotools](https://github.com/pdollar/coco/tree/master/PythonAPI/pycocotools), OpenCV.
+ Pre-trained [ResNet model](http://models.tensorpack.com/ResNet/) from tensorpack model zoo.
+ COCO data. It assumes the following directory structure:
```
DIR/
  annotations/
    instances_train2014.json
    instances_val2014.json
    instances_minival2014.json
    instances_valminusminival2014.json
  train2014/
    COCO_train2014_*.jpg
  val2014/
    COCO_val2014_*.jpg
```
`minival` and `valminusminival` are optional. You can download them
[here](https://github.com/rbgirshick/py-faster-rcnn/blob/master/data/README.md).


## Usage
Change config in `config.py`:
1. Change `BASEDIR` to `/path/to/DIR` as described above.
2. Change `MODE_MASK` to switch Faster-RCNN or Mask-RCNN.

Train:
```
./train.py --load /path/to/ImageNet-ResNet50.npz
```
The code is only for training with 1, 2, 4 or 8 GPUs.
Otherwise, you probably need different hyperparameters for the same performance.

Predict on an image (and show output in a window):
```
./train.py --predict input.jpg --load /path/to/model
```

Evaluate the performance of a model and save to json.
(Pretrained models can be downloaded in [model zoo](http://models.tensorpack.com/FasterRCNN):
```
./train.py --evaluate output.json --load /path/to/model
```

## Results

These models are trained with different configurations on trainval35k and evaluated on minival using mAP@IoU=0.50:0.95.
MaskRCNN results contain both bbox and segm mAP.

|Backbone|`FASTRCNN_BATCH`|resolution |schedule|mAP (bbox/segm)|Time         |
|   -    |    -           |    -      |   -    |   -           |   -         |
|R-50    |64              |(600, 1024)|280k    |33.0           |22h on 8 P100|
|R-50    |512             |(800, 1333)|280k    |35.6           |55h on 8 P100|
|R-50    |512             |(800, 1333)|360k    |36.7           |49h on 8 V100|
|R-50    |256             |(800, 1333)|280k    |36.9/32.3      |39h on 8 P100|
|R-50    |512							|(800, 1333)|360k    |37.7/33.0      |72h on 8 P100|
|R-101   |512             |(800, 1333)|280k    |40.1/34.4      |70h on 8 P100|

The two 360k models have identical configurations with
`R50-C4-2x` configuration in
[Detectron Model Zoo](https://github.com/facebookresearch/Detectron/blob/master/MODEL_ZOO.md#end-to-end-faster--mask-r-cnn-baselines).
They get the __same performance__ with the official models, and are about 14% slower than the official implementation,
probably due to the lack of specialized ops (e.g. AffineChannel, ROIAlign) in TensorFlow.

## Notes

See [Notes on This Implementation](NOTES.md)
