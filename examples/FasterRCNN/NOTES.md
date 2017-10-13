
### File Structure
This is a minimal implementation that simply contains these files:
+ coco.py: load COCO data
+ data.py: prepare data for training
+ common.py: common data preparation utilities
+ basemodel.py: implement resnet
+ model.py: implement rpn/faster-rcnn
+ train.py: main training script
+ utils/: third-party helper functions
+ eval.py: evaluation utilities
+ viz.py: visualization utilities

### Implementation Notes

1. You can easily add more augmentations such as rotation, but be careful how a box should be
	 augmented. The code now will always use the minimal axis-aligned bounding box of the 4 corners,
	 which is probably not the optimal way.

2. Floating-point boxes are defined like this:

<p align="center"> <img src="https://user-images.githubusercontent.com/1381301/31527740-2f1b38ce-af84-11e7-8de1-628e90089826.png"> </p>

3. Inference is not quite fast, because either you disable convolution autotune and end up with
	 a slow convolution algorithm, or you spend more time on autotune.
	 This is a general problem of TensorFlow when running against variable-sized input.

4. In Faster-RCNN, BatchNorm statistics are not supposed to be updated during fine-tuning.
	 This specific kind of BatchNorm will need [my kernel](https://github.com/tensorflow/tensorflow/pull/12580)
	 which is included since TF 1.4. If using an earlier version of TF, it will be either slow or wrong.
