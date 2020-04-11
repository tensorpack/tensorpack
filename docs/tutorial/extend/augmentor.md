

### Write an Image Augmentor

The first thing to note: __you never have to write an augmentor__.
An augmentor is a part of the DataFlow, so you can always
[write a DataFlow](./dataflow.md)
to do whatever operations to your data, rather than writing an augmentor.

Augmentor makes things easier when what you want fits its design.
But remember it is just an abstraction that may not always work for your use case.
For example, if your data transformation depend on multiple dataflow components,
or if you want to apply different transformations to different components,
the abstraction is often not enough for you, and you need to write code on the
DataFlow level instead.

An image augmentor maps an image to an image.
If you have such a mapping function `f` already, you can simply use
[imgaug.MapImage(f)](../../modules/dataflow.imgaug.html#tensorpack.dataflow.imgaug.MapImage)
as the augmentor, or use
[MapDataComponent(dataflow, f, index)](../../modules/dataflow.html#tensorpack.dataflow.MapDataComponent)
as the DataFlow.
In other words, for simple mapping you do not need to write an augmentor.

An augmentor may do something more than just applying a mapping.
To do custom augmentation, you can implement one yourself.


#### The Design of imgaug Module

The [imgaug module](../../modules/dataflow.imgaug) is designed to allow the following usage:

* Factor out randomness and determinism.
  An augmentor often contains randomized policy, e.g., it randomly perturbs each image differently.
  However, its "deterministic" part needs to be factored out, so that
  the same transformation can be re-applied to other data
  assocaited with the image. This is achieved like this:

```python
tfm = augmentor.get_transform(img)  # a deterministic transformation
new_img = tfm.apply_image(img)
new_img2 = tfm.apply_image(img2)
new_coords = tfm.apply_coords(coords)
```

  Due to this design, it can augment images together with its annotations
  (e.g., segmentation masks, bounding boxes, keypoints).
  Our coordinate augmentation enforces floating points coordinates
  to avoid quantization error.

  When you don't need to re-apply the same transformation, you can also just call

```python
new_img = augmentor.augment(img)
```

* Reset random seed. Random seed can be reset by
  [reset_state](../../modules/dataflow.imgaug.html#tensorpack.dataflow.imgaug.ImageAugmentor.reset_state).
  This is important for multi-process data loading, to make sure different
  processes get different seeds.
  The reset method is called automatically if you use tensorpack's
  [image augmentation dataflow](../../modules/dataflow.html#tensorpack.dataflow.AugmentImageComponent)
  or if you use Python 3.7+.
  Otherwise, **you are responsible** for calling it by yourself in subprocesses.
  See the
  [API documentation](../../modules/dataflow.imgaug.html#tensorpack.dataflow.imgaug.ImageAugmentor.reset_state)
  of this method for more details.


### Write an Augmentor

The interface you will need to implement is:

```python
class MyAug(imgaug.ImageAugmentor):
  def get_transform(self, img):
    # Randomly generate a deterministic transformation, to be applied on img
    x = random_parameters()
    return MyTransform(x)

class MyTransform(imgaug.Transform):
  def apply_image(self, img):
    return new_img

  def apply_coords(self, coords):
    return new_coords
```

Check out the zoo of builtin augmentors to have a better sense.
