## Balloon Demo

This is a demo on how to train tensorpack's Mask R-CNN on a custom dataset.
We use the [balloon dataset](https://github.com/matterport/Mask_RCNN/tree/master/samples/balloon)
as an example.

1. Download and unzip the dataset:
```
wget https://github.com/matterport/Mask_RCNN/releases/download/v2.1/balloon_dataset.zip
unzip balloon_dataset.zip
```

2. (included already) Since this dataset is not in COCO format, we add a new file
	 [dataset/balloon.py](dataset/balloon.py) to load the dataset.
	 Refer to [dataset/dataset.py](dataset/dataset.py) on the required interface of a new dataset.

3. (included already) Register the names of the new dataset in `train.py` and `predict.py`, by calling `register_balloon("/path/to/balloon_dataset")`

4. Download a model pretrained on COCO from tensorpack model zoo:
```
wget http://models.tensorpack.com/FasterRCNN/COCO-MaskRCNN-R50FPN2x.npz
```

5. Start fine-tuning on the new dataset:
```
./train.py --config DATA.BASEDIR=~/data/balloon MODE_FPN=True \
	"DATA.VAL=('balloon_val',)"  "DATA.TRAIN=('balloon_train',)" \
	TRAIN.BASE_LR=1e-3 TRAIN.EVAL_PERIOD=0 "TRAIN.LR_SCHEDULE=[1000]" \
	"PREPROC.TRAIN_SHORT_EDGE_SIZE=[600,1200]" TRAIN.CHECKPOINT_PERIOD=1 DATA.NUM_WORKERS=1 \
	--load COCO-MaskRCNN-R50FPN2x.npz --logdir train_log/balloon
```

6. You can train as long as you want, but it only takes __a few minutes__ to produce nice results.
  You can visualize the results of the latest model by:
```
./predict.py --config DATA.BASEDIR=~/data/balloon MODE_FPN=True \
	"DATA.VAL=('balloon_val',)"  "DATA.TRAIN=('balloon_train',)" \
	--load train_log/balloon/checkpoint --predict ~/data/balloon/val/*.jpg
```

This command will produce images like this in your window:

![demo](https://user-images.githubusercontent.com/1381301/62665002-915ff880-b932-11e9-9f7e-f24f83d5d69c.jpg)


