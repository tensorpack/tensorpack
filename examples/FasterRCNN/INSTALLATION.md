# Faster R-CNN with Feature Pyramid Network (FPN)

## Reference
https://arxiv.org/abs/1506.01497
https://arxiv.org/abs/1612.03144

# Implementation
https://github.com/tensorpack/tensorpack with commit id 79148350eabd6800133a49b101eea4c56e78e4e8

## Data Set
COCO 2017 (http://cocodataset.org/)

## Validation Command(s)
From the root directory of the above implementation, set the python path using the following command.
```
export PYTHONPATH=$PYTHONPATH:`pwd`
```

From the root directory of the above implementation, change to the Faster RCNN directory using the following command.
```
cd ./examples/FasterRCNN/
```

From the `FasterRCNN` directory of the above implementation, create a `temp` directory using the following command.
```
mkdir ./temp
mkdir ./temp/val2014
```

From the `FasterRCNN` directory of the above implementation, download the validation images file to the `./temp` directory using the following command.
```
wget http://images.cocodataset.org/zips/val2017.zip -P ./temp/
```

From the `FasterRCNN` directory of the above implementation, decompress the validation images file using the following command.
```
unzip ./temp/val2017.zip -d ./temp/
```

From the `FasterRCNN` directory of the above implementation, download the validation annotations file to the `./temp` directory using the following command.
```
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip -P ./temp/
```

From the `FasterRCNN` directory of the above implementation, decompress the validation annotations file using the following command.
```
unzip ./temp/annotations_trainval2017.zip -d ./temp/
```

From the `FasterRCNN` directory of the above implementation, copy the model files to `./temp` using the following command.
```
cp /nfs/fm/disks/aipg_trained_models_01/tensorflow/faster-rcnn-fpn/coco/* ./temp/
```

From the `FasterRCNN` directory of the above implementation, validate the model using the following command.
```
python train.py --evaluate ./temp/output.json --load ./temp/model.ckpt --config MODE_MASK=False MODE_FPN=True DATA.BASEDIR=./temp/ DATA.VAL='val2017' TRAIN.NUM_GPUS=1
```
Upon completion of this command, the validation COCO mAP (`Average Precision (AP)@[IoU=0.50:0.95 | area=all | maxDets=100]`) will be displayed and should closely match the validation COCO mAP shown below.

## Validation COCO mAP
37.3%