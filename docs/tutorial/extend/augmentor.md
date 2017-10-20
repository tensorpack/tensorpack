
### Write an Image Augmentor

The first thing to note: __you never have to write an augmentor__.
An augmentor is a part of the DataFlow, so you can always
[write a DataFlow](dataflow.html)
to do whatever operations to your data, rather than writing an augmentor.
Augmentors just sometimes make things easier.

An image augmentor maps an image to an image.
If you have such a mapping function `f` already, you can simply use `imgaug.MapImage(f)` as the
augmentor, or use `MapDataComponent(dataflow, f, index)` as the DataFlow.
In other words, for simple mapping you do not need to write an augmentor.

An augmentor may do something more than just applying a mapping.
The interface you will need to implement is:

```python
class MyAug(imgaug.ImageAugmentor):
  def _get_augment_params(self, img):
    # generated random params with self.rng
    return params

  def _augment(self, img, params):
    return augmented_img

  # optional method
  def _augment_coords(self, coords, param):
    # coords is a Nx2 floating point array, each row is (x, y)
    return augmented_coords
```

It does the following extra things for you:

1. `self.rng` is a `np.random.RandomState` object,
  guaranteed to have different seeds when you use multiprocess prefetch.
  In multiprocess settings, you have to use this rng to generate random numbers.

2. The logic of random parameter generation and the actual augmentation is separated in different methods.
  This allows you to apply the
  same transformation to several images together (with `AugmentImageComponents`),
  which is essential to tasks such as segmentation.
  Or apply the same transformations to images plus coordinate labels (with `AugmentImageCoordinates`),
  which is essential to tasks such as detection and localization.
