tensorpack.dataflow.imgaug package
==================================

This package contains Tensorpack's augmentors.
The imgaug module is designed to allow the following usage:

1. Factor out randomness and determinism.
   An augmentor may be randomized, but you can call
   `augment_return_params <#tensorpack.dataflow.imgaug.Augmentor.augment_return_params>`_
   to obtain the randomized parameters and then call
   `augment_with_params <#tensorpack.dataflow.imgaug.Augmentor.augment_with_params>`_
   on other data with the same randomized parameters.

2. Because of (1), tensorpack's augmentor can augment multiple images together
   easily. This is commonly used for augmenting an image together with its masks.

3. An image augmentor (e.g. flip) may also augment a coordinate, with
   `augment_coords <#tensorpack.dataflow.imgaug.ImageAugmentor.augment_coords>`_.
   In this way, images can be augmented together with
   boxes, polygons, keypoints, etc.
   Coordinate augmentation enforces floating points coordinates
   to avoid quantization error.

4. Reset random seed. Random seed can be reset by
   `reset_state <#tensorpack.dataflow.imgaug.Augmentor.reset_state>`_.
   This is important for multi-process data loading, and
   it is called automatically if you use tensorpack's
   `image augmentation dataflow <dataflow.html#tensorpack.dataflow.AugmentImageComponent>`_.

Note that other image augmentation libraries can be wrapped into Tensorpack's interface as well.
For example, `imgaug.IAAugmentor <#tensorpack.dataflow.imgaug.IAAugmentor>`_
and `imgaug.Albumentations <#tensorpack.dataflow.imgaug.Albumentations>`_
wrap two popular image augmentation libraries.


.. container:: custom-index

    .. raw:: html

        <script type="text/javascript" src='../_static/build_toc_group.js'></script>

.. automodule:: tensorpack.dataflow.imgaug
    :members:
    :undoc-members:
    :show-inheritance:
