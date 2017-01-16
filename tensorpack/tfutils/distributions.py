import tensorflow as tf
from functools import wraps
import numpy as np

__all__ = ['CategoricalDistribution']


def class_scope(method):
    """Enhance TensorBoard graph visualization by grouping operators.

    If class of method has member "name", then this is used for name-scoping.

    Args:
        method (TYPE): method from a python class

    Returns:
        TYPE: Description
    """
    @wraps(method)
    def _impl(self, *method_args, **method_kwargs):
        # is there specific name_scope?
        if hasattr(self, "name"):
            distr_name = self.name
        else:
            distr_name = self.__class__.__name__
        # is not already scoped with current class to prevent nested scopes due to nested methods
        if distr_name not in tf.no_op(name='.').name[:-1]:
            with tf.name_scope(distr_name + "_" + method.__name__):
                return method(self, *method_args, **method_kwargs)
        else:
            return method(self, *method_args, **method_kwargs)
    return _impl


class Distribution(object):
    """Represent a distribution.
    """

    def __init__(self, name):
        self.name = name

    def loglikelihood(self, x, theta):
        return self._loglikelihood(x, theta)

    @class_scope
    def mutual_information(self, x, theta):
        expected_log_qc = self.entropy(x)
        expected_log_qc_given_x = self.cross_entropy(x, theta)
        return tf.add(expected_log_qc, expected_log_qc_given_x, name="mutual_information")

    @class_scope
    def entropy(self, x):
        return tf.reduce_mean(self.loglikelihood_prior(x), name="entropy")

    @class_scope
    def cross_entropy(self, x, theta):
        return tf.reduce_mean(self.loglikelihood(x, theta), name="cross_entropy")

    @class_scope
    def loglikelihood_prior(self, x):
        theta = self._prior()
        return self._loglikelihood(x, theta)

    @class_scope
    def prior(self):
        return self._prior()

    @class_scope
    def code(self, batch_size, name="zc"):
        return self._code(batch_size, name)

    def _loglikelihood(self, x, theta):
        raise NotImplementedError

    def _prior(self):
        raise NotImplementedError

    def _code(self, batch_size, name="zc"):
        raise NotImplementedError


class CategoricalDistribution(Distribution):
    def __init__(self, name, cardinality):
        super(CategoricalDistribution, self).__init__(name)
        self.cardinality = cardinality

    def _loglikelihood(self, x, theta):
        eps = 1e-8
        return tf.reduce_sum(tf.log(theta + eps) * x, reduction_indices=1)

    def _prior(self):
        return tf.constant([1.0 / self.cardinality] * self.cardinality)

    def _code(self, batch_size, name="zc"):
        ids = tf.multinomial(tf.zeros([batch_size, self.cardinality]), num_samples=1)[:, 0]
        zc = tf.one_hot(ids, self.cardinality)
        zc = tf.placeholder_with_default(zc, [None, self.cardinality], name=name)
        return zc


class UniformDistribution(Distribution):
    def __init__(self, name, dim):
        super(UniformDistribution, self).__init__(name)
        self.dim = dim

    def _loglikelihood(self, x, theta):
        eps = 1e-8

        l = theta.get_shape()[1]
        mean = theta[:, :l, :]
        stddev = theta[:, :l, :]

        exponent = (x - mean) / (stddev + eps)

        return tf.reduce_sum(
            - 0.5 * np.log(2 * np.pi) - tf.log(stddev + eps) - 0.5 * tf.square(exponent),
            reduction_indices=1,
        )

    def _prior(self):
        return tf.random_uniform([self.dim], minval=-1., maxval=1.)

    def _code(self, batch_size, name="zc"):
        zc = tf.random_uniform([batch_size, self.dim], -1, 1)
        zc = tf.placeholder_with_default(zc, [None, self.dim], name=name)
        return zc
