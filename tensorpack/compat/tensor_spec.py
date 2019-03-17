
"""
Copied from tensorflow/python/framework/tensor_spec.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.framework import common_shapes
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape


class TensorSpec(object):
  """Describes a tf.Tensor.

  Metadata for describing the `tf.Tensor` objects accepted or returned
  by some TensorFlow APIs.
  """

  __slots__ = ["_shape", "_shape_tuple", "_dtype", "_name"]

  def __init__(self, shape, dtype=dtypes.float32, name=None):
    """Creates a TensorSpec.

    Args:
      shape: Value convertible to `tf.TensorShape`. The shape of the tensor.
      dtype: Value convertible to `tf.DType`. The type of the tensor values.
      name: Optional name for the Tensor.

    Raises:
      TypeError: If shape is not convertible to a `tf.TensorShape`, or dtype is
        not convertible to a `tf.DType`.
    """
    self._shape = tensor_shape.TensorShape(shape)
    try:
      self._shape_tuple = tuple(self.shape.as_list())
    except ValueError:
      self._shape_tuple = None
    self._dtype = dtypes.as_dtype(dtype)
    self._name = name

  @classmethod
  def from_spec(cls, spec, name=None):
    return cls(spec.shape, spec.dtype, name or spec.name)

  @classmethod
  def from_tensor(cls, tensor, name=None):
    if isinstance(tensor, ops.EagerTensor):
      return TensorSpec(tensor.shape, tensor.dtype, name)
    elif isinstance(tensor, ops.Tensor):
      return TensorSpec(tensor.shape, tensor.dtype, name or tensor.op.name)
    else:
      raise ValueError("`tensor` should be a tf.Tensor")

  @property
  def shape(self):
    """Returns the `TensorShape` that represents the shape of the tensor."""
    return self._shape

  @property
  def dtype(self):
    """Returns the `dtype` of elements in the tensor."""
    return self._dtype

  @property
  def name(self):
    """Returns the (optionally provided) name of the described tensor."""
    return self._name

  def is_compatible_with(self, spec_or_tensor):
    """Returns True if spec_or_tensor is compatible with this TensorSpec.

    Two tensors are considered compatible if they have the same dtype
    and their shapes are compatible (see `tf.TensorShape.is_compatible_with`).

    Args:
      spec_or_tensor: A tf.TensorSpec or a tf.Tensor

    Returns:
      True if spec_or_tensor is compatible with self.
    """
    return (self._dtype.is_compatible_with(spec_or_tensor.dtype) and
            self._shape.is_compatible_with(spec_or_tensor.shape))

  def __repr__(self):
    return "TensorSpec(shape={}, dtype={}, name={})".format(
        self.shape, repr(self.dtype), repr(self.name))

  def __hash__(self):
    return hash((self._shape_tuple, self.dtype))

  def __eq__(self, other):
    return (self._shape_tuple == other._shape_tuple  # pylint: disable=protected-access
            and self.dtype == other.dtype
            and self._name == other._name)  # pylint: disable=protected-access

  def __ne__(self, other):
    return not self == other

  def __reduce__(self):
    return TensorSpec, (self._shape, self._dtype, self._name)
