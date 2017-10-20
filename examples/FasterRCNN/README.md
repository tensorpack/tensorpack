# Faster-RCNN on COCO
This example aims to provide a minimal (<1000 lines) multi-GPU implementation of ResNet50-Faster-RCNN on COCO.

## Dependencies
+ TensorFlow >= 1.4.0rc0
+ Install [pycocotools](https://github.com/pdollar/coco/tree/master/PythonAPI/pycocotools), OpenCV.
+ Pre-trained [ResNet50 model](https://goo.gl/6XjK9V) from tensorpack model zoo.
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
Change `BASEDIR` in `config.py` to `/path/to/DIR` as described above.

To train:
```
./train.py --load /path/to/ImageNet-ResNet50.npz
```
The code is written for training with __8 GPUs__.
To run on fewer GPUs, you probably need different hyperparameters for the same performance.

To predict on an image (and show output in a window):
```
./train.py --predict input.jpg
```

## Results

+ trainval35k/minival, FASTRCNN_BATCH=256: 32.9
+ trainval35k/minival, FASTRCNN_BATCH=64: 31.6. Takes less than one day on 8 Maxwell TitanX.

The hyperparameters are not carefully tuned. You can probably get better performance by e.g.  training longer.

## Notes

See [Notes on This Implementation](NOTES.md)
