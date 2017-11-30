# Faster-RCNN / Mask-RCNN on COCO
This example aims to provide a minimal (1.3k lines) multi-GPU implementation of
Faster-RCNN / Mask-RCNN (without FPN) on COCO.

## Dependencies
+ Python 3; TensorFlow >= 1.4.0
+ Install [pycocotools](https://github.com/pdollar/coco/tree/master/PythonAPI/pycocotools), OpenCV.
+ Pre-trained [ResNet model](https://goo.gl/6XjK9V) from tensorpack model zoo.
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
1. Set `BASEDIR` to `/path/to/DIR` as described above.
2. Set `MODE_MASK` to switch Faster-RCNN or Mask-RCNN.

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

Models are trained on trainval35k and evaluated on minival using mAP@IoU=0.50:0.95.
MaskRCNN results contain both bbox and segm mAP.

|Backbone | `FASTRCNN_BATCH` | resolution | mAP (bbox/segm) | Time |
| - | - | - | - | - |
| Res50 | 64 | (600, 1024) | 33.0 | 22h on 8 P100 |
| Res50 | 256 | (600, 1024) | 34.4 | 49h on 8 M40 |
| Res50 | 512 | (800, 1333) | 35.6 | 55h on 8 P100|
| Res50 | 512 | (800, 1333) | 36.9/32.3 | 59h on 8 P100|
| Res101 | 512 | (800, 1333) | 40.1/34.4 | 70h on 8 P100|

Note that these models are trained with a larger ROI batch size than the paper,
and get about 1mAP better performance.

## Notes

See [Notes on This Implementation](NOTES.md)
