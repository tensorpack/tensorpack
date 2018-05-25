# Faster-RCNN / Mask-RCNN on COCO
This example provides a minimal (only 1.6k lines) but faithful implementation of
the following papers:

+ [Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](https://arxiv.org/abs/1506.01497)
+ [Feature Pyramid Networks for Object Detection](https://arxiv.org/abs/1612.03144)
+ [Mask R-CNN](https://arxiv.org/abs/1703.06870)

## Dependencies
+ Python 3; TensorFlow >= 1.4.0 (>=1.6.0 recommended due to a TF bug);
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
2. Change `MODE_MASK`/`MODE_FPN`, or other options you like. Recommended configurations are listed in the table below.

Train:
```
./train.py --load /path/to/ImageNet-ResNet50.npz
```
The code is only valid for training with 1, 2, 4 or 8 GPUs.
Not training with 8 GPUs may result in different performance from the table below.

Predict on an image (and show output in a window):
```
./train.py --predict input.jpg --load /path/to/model
```

Evaluate the performance of a model on COCO, and save results to json.
(Pretrained models can be downloaded in [model zoo](http://models.tensorpack.com/FasterRCNN):
```
./train.py --evaluate output.json --load /path/to/model
```

## Results

These models are trained with different configurations on trainval35k and evaluated on minival using mAP@IoU=0.50:0.95.
MaskRCNN results contain both bbox and segm mAP.

|Backbone|`FASTRCNN_BATCH`|resolution |schedule|mAP (bbox/segm)|Time          |
|   -    |    -           |    -      |   -    |   -           |   -          |
|R50-C4  |64              |(600, 1024)|280k    |33.1           |18h on 8 V100s|
|R50-C4  |512             |(800, 1333)|280k    |35.6           |55h on 8 P100s|
|R50-C4  |512             |(800, 1333)|360k    |36.6           |49h on 8 V100s|
|R50-FPN |512             |(800, 1333)|360k    |37.5           |28h on 8 V100s|
|R50-C4  |256             |(800, 1333)|280k    |36.8/32.1      |39h on 8 P100s|
|R50-C4  |512             |(800, 1333)|360k    |37.8/33.1      |51h on 8 V100s|
|R50-FPN |512             |(800, 1333)|360k    |38.1/34.9      |38h on 8 V100s|
|R101-C4 |512             |(800, 1333)|280k    |40.1/34.4      |70h on 8 P100s|
|R101-C4 |512             |(800, 1333)|360k    |40.8/35.1      |63h on 8 V100s|

The two R50-C4 360k models have the same configuration __and mAP__
as the `R50-C4-2x` entries in
[Detectron Model Zoo](https://github.com/facebookresearch/Detectron/blob/master/MODEL_ZOO.md#end-to-end-faster--mask-r-cnn-baselines).
So far this is the only public TensorFlow implementation that can reproduce mAP in Detectron.
The other models listed here do not correspond to any configurations in Detectron.

## Notes

See [Notes on This Implementation](NOTES.md)
