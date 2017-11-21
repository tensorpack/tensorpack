# Faster-RCNN on COCO
This example aims to provide a minimal (1.2k lines) multi-GPU implementation of ResNet-Faster-RCNN on COCO.

## Dependencies
+ TensorFlow >= 1.4.0
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
The code is only for training with 1, 2, 4 or 8 GPUs.
Otherwise, you probably need different hyperparameters for the same performance.

To predict on an image (and show output in a window):
```
./train.py --predict input.jpg --load /path/to/model
```

To evaluate the performance (pretrained models can be downloaded in [model zoo](https://drive.google.com/open?id=1J0xuDAuyOWiuJRm2LfGoz5PUv9_dKuxq):
```
./train.py --evaluate output.json --load /path/to/model
```

## Results

Mean Average Precision @IoU=0.50:0.95:

+ trainval35k/minival, FASTRCNN_BATCH=256: 34.2. Takes 49h on 8 TitanX.
+ trainval35k/minival, FASTRCNN_BATCH=64: 33.0. Takes 22h on 8 P100.

The hyperparameters are not carefully tuned. You can probably get better performance by e.g. training longer.

## Notes

See [Notes on This Implementation](NOTES.md)
