
### File Structure
This is a minimal implementation that simply contains these files:
+ train.py,predict.py: main entry script
+ modeling/generalized_rcnn.py: implement variants of generalized R-CNN architecture
+ modeling/backbone.py: implement backbones
+ modeling/model_{fpn,rpn,frcnn,mrcnn,cascade}.py: implement FPN,RPN,Fast/Mask/Cascade R-CNN models.
+ modeling/model_box.py: implement box-related symbolic functions
+ dataset/dataset.py: the dataset interface
+ dataset/coco.py: load COCO data to the dataset interface
+ data.py: prepare data for training & inference
+ common.py: common data preparation utilities
+ utils/: third-party helper functions
+ eval.py: evaluation utilities
+ viz.py: visualization utilities

### Implementation Notes

#### Data:

1. It's easy to train on your own data, by calling `DatasetRegistry.register(name, lambda: YourDatasetSplit())`,
	 and modify `cfg.DATA.*` accordingly. Afterwards, "name" can be used in `cfg.DATA.TRAIN`.

	`YourDatasetSplit` can be:

   + `COCODetection`, if your data is already in COCO format. In this case, you need to
		 modify `dataset/coco.py` to change the class names and the id mapping.

   + Your own class, if your data is not in COCO format.
		 You need to write a subclass of `DatasetSplit`, similar to `COCODetection`.
     In this class you'll implement the logic to load your dataset and evaluate predictions.
		 The documentation is in the docstring of `DatasetSplit.

	 See [BALLOON.md](BALLOON.md) for an example of fine-tuning on a different dataset.

1. You can easily add more augmentations such as rotation, but be careful how a box should be
   augmented. The code now will always use the minimal axis-aligned bounding box of the 4 corners,
   which is probably not the optimal way.
   A TODO is to generate bounding box from segmentation, so more augmentations can be naturally supported.

#### Model:

1. Floating-point boxes are defined like this:

<p align="center"> <img src="https://user-images.githubusercontent.com/1381301/31527740-2f1b38ce-af84-11e7-8de1-628e90089826.png"> </p>

2. We use ROIAlign, and `tf.image.crop_and_resize` is __NOT__ ROIAlign.

3. We currently only support single image per GPU in this example.

4. Because of (3), BatchNorm statistics are supposed to be freezed during fine-tuning.

5. An alternative to freezing BatchNorm is to sync BatchNorm statistics across
   GPUs (the `BACKBONE.NORM=SyncBN` option).
   Another alternative to BatchNorm is GroupNorm (`BACKBONE.NORM=GN`) which has better performance.

#### Efficiency:

Training throughput (larger is better) of standard R50-FPN Mask R-CNN, on 8 V100s:

| Implementation                                                                                    | Throughput (img/s) |
|---------------------------------------------------------------------------------------------------|:------------------:|
| [Detectron2](https://github.com/facebookresearch/detectron2)                                      | 60                 |
| [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark/)                     | 51                 |
| tensorpack                                                                                        | 50                 |
| [mmdetection](https://github.com/open-mmlab/mmdetection/blob/master/docs/MODEL_ZOO.md#mask-r-cnn) | 41                 |
| [Detectron](https://github.com/facebookresearch/Detectron)                                        | 19                 |
| [matterport/Mask_RCNN](https://github.com/matterport/Mask_RCNN/)                                  | 14                 |

1. This implementation does not use specialized CUDA ops (e.g. ROIAlign),
   and does not use batch of images.
   Therefore it might be slower than other highly-optimized implementations.
	 For details of the benchmark, see [detectron2 benchmarks](https://detectron2.readthedocs.io/notes/benchmarks.html).

1. If CuDNN warmup is on, the training will start very slowly, until about
   10k steps (or more if scale augmentation is used) to reach a maximum speed.
   As a result, the ETA is also inaccurate at the beginning.
   CuDNN warmup is by default enabled when no scale augmentation is used.

1. After warmup, the training speed will slowly decrease due to more accurate proposals.

1. The code should have around 85~90% GPU utilization on one V100.
	Scalability isn't very meaningful since the amount of computation each GPU perform is data-dependent.
	If all images have the same spatial size (in which case the per-GPU computation is *still different*),
	then a 85%~90% scaling efficiency is observed when using 8 V100s and `HorovodTrainer`.

1. To reduce RAM usage on host: (1) make sure you're using the "spawn" method as
   set in `train.py`; (2) reduce `buffer_size` or `NUM_WORKERS` in `data.py`
   (which may negatively impact your throughput). The training only needs <10G RAM if `NUM_WORKERS=0`.

1. Inference is unoptimized. Tensorpack is a training interface: it produces the trained weights
	 in standard format but it does not help you on optimized inference.
	 In fact, the current implementation uses some slow numpy operations in inference (in `eval.py:_paste_mask`).

Possible Future Enhancements:

1. Support batch>1 per GPU. Batching with inconsistent shapes is
   non-trivial to implement in TensorFlow.

1. Use dedicated ops to improve speed. (e.g. a TF implementation of ROIAlign op
   can be found in [light-head RCNN](https://github.com/zengarden/light_head_rcnn/tree/master/lib/lib_kernel))


### TensorFlow version notes

TensorFlow ≥ 1.6 supports most common features in this R-CNN implementation.
However, each version of TensorFlow has bugs that I either reported or fixed,
and this implementation touches many of those bugs.
Therefore, not every version of TF ≥ 1.6 supports every feature in this implementation.

1. TF < 1.6: Nothing works due to lack of support for empty tensors
   ([PR](https://github.com/tensorflow/tensorflow/pull/15264))
   and `FrozenBN` training
   ([PR](https://github.com/tensorflow/tensorflow/pull/12580)).
1. TF < 1.10: `SyncBN` with NCCL will fail ([PR](https://github.com/tensorflow/tensorflow/pull/20360)).
1. TF 1.11 & 1.12: multithread inference will fail ([issue](https://github.com/tensorflow/tensorflow/issues/22750)).
   Latest tensorpack will apply a workaround.
1. TF 1.13: MKL inference will fail ([issue](https://github.com/tensorflow/tensorflow/issues/24650)).
1. TF > 1.12: Horovod training will fail ([issue](https://github.com/tensorflow/tensorflow/issues/25946)).
   Latest tensorpack will apply a workaround.

This implementation contains workaround for some of these TF bugs.
However, note that the workaround needs to check your TF version by `tf.VERSION`,
and may not detect bugs properly if your TF version is not an official release
(e.g., if you use a nightly build).
