import tensorflow as tf
from functools import wraps
import numpy as np
from ..tfutils import get_name_scope_name

__all__ = ['Distribution',
           'CategoricalDistribution', 'GaussianDistribution',
           'ProductDistribution']


def class_scope(func):
    """
    A decorator which wraps a function with a name_scope: "{class_name}_{method_name}".
    The "{class_name}" is either ``cls.name`` or simply the class name.
    It helps enhance TensorBoard graph visualization by grouping operators.

    This is just syntactic sugar to prevent writing: with
    ``tf.name_scope(...)`` in each method.
    """

    @wraps(func)
    def _impl(self, *args, **kwargs):
        # is there a specific name?
        distr_name = self.name
        if distr_name is None:
            distr_name = self.__class__.__name__
        # scope it only when it is not already scoped with current class
        if distr_name not in get_name_scope_name():
            with tf.name_scope(distr_name + "_" + func.__name__):
                return func(self, *args, **kwargs)
        else:
            return func(self, *args, **kwargs)
    return _impl


class Distribution(object):
    """
    Base class of symbolic distribution utilities
    (the distribution parameters can be symbolic tensors).
    """

    name = None

    def __init__(self, name):
        """
        Args:
            name(str): the name to be used for scope and tensors in this
                distribution.
        """
        self.name = name

    @class_scope
    def loglikelihood(self, x, theta):
        """
        Args:
            x: samples of shape (batch, sample_dim)
            theta: model parameters of shape (batch, param_dim)

        Returns:
            log likelihood of each sample, of shape (batch,)
        """
        assert x.get_shape().ndims == 2 and \
            x.get_shape()[1] == self.sample_dim, \
            x.get_shape()
        assert theta.get_shape().ndims == 2 and \
            theta.get_shape()[1] == self.param_dim, \
            theta.get_shape()

        ret = self._loglikelihood(x, theta)
        assert ret.get_shape().ndims == 1, ret.get_shape()
        return ret

    @class_scope
    def entropy(self, x, theta):
        r""" Entropy of this distribution parameterized by theta,
            estimated from a batch of samples.

        .. math::

            H(x) = - E[\log p(x_i)], \text{where } p \text{ is parameterized by } \theta.

        Args:
            x: samples of shape (batch, sample_dim)
            theta: model parameters of shape (batch, param_dim)

        Returns:
            a scalar tensor, the entropy.
        """
        return tf.reduce_mean(-self.loglikelihood(x, theta), name="entropy")

    @class_scope
    def sample(self, batch_size, theta):
        """
        Sample a batch of vectors from this distribution parameterized by theta.

        Args:
            batch_size(int): the batch size.
            theta: a tensor of shape (param_dim,) or (batch, param_dim).

        Returns:
            a batch of samples of shape (batch, sample_dim)
        """
        assert isinstance(batch_size, int), batch_size
        shp = theta.get_shape()
        assert shp.ndims in [1, 2] and shp[-1] == self.sample_dim, shp
        if shp.ndims == 1:
            theta = tf.tile(tf.expand_dims(theta, 0), [batch_size, 1],
                            name='tiled_theta')
        else:
            assert shp[0] == batch_size, shp
        x = self._sample(batch_size, theta)
        assert x.get_shape().ndims == 2 and \
            x.get_shape()[1] == self.sample_dim, \
            x.get_shape()
        return x

    @class_scope
    def encoder_activation(self, dist_param):
        """ An activation function which transform unconstrained raw network output
            to a vector of feasible distribution parameters.

            Note that for each distribution,
            there are many feasible ways to design this function and it's hard to say which is better.
            The default implementations in the distribution classes here is
            just one reasonable way to do this.

        Args:
            dist_param: output from a network, of shape (batch, param_dim).

        Returns:
            a tensor of the same shape, the distribution parameters.
        """
        return self._encoder_activation(dist_param)

    @property
    def param_dim(self):
        """
        Returns:
            int: the dimension of parameters of this distribution.
        """
        raise NotImplementedError()

    @property
    def sample_dim(self):
        """
        Returns:
            int: the dimension of samples out of this distribution.
        """
        raise NotImplementedError()

    def _loglikelihood(self, x, theta):
        raise NotImplementedError()

    def _encoder_activation(self, dist_param):
        return dist_param

    def _sample(self, batch_size, theta):
        raise NotImplementedError()


class CategoricalDistribution(Distribution):
    """ Categorical distribution of a set of classes.
        Each sample is a one-hot vector.
    """
    def __init__(self, name, cardinality):
        """
        Args:
            cardinality (int): number of categories
        """
        super(CategoricalDistribution, self).__init__(name)
        self.cardinality = cardinality

    def _loglikelihood(self, x, theta):
        eps = 1e-8
        return tf.reduce_sum(tf.log(theta + eps) * x, 1)

    def _encoder_activation(self, dist_param):
        return tf.nn.softmax(dist_param)

    def _sample(self, batch_size, theta):
        ids = tf.squeeze(tf.multinomial(
            tf.log(theta + 1e-8), num_samples=1), 1)
        return tf.one_hot(ids, self.cardinality, name='sample')

    @property
    def param_dim(self):
        return self.cardinality

    @property
    def sample_dim(self):
        return self.cardinality


class GaussianDistribution(Distribution):
    def __init__(self, name, dim, fixed_std=True):
        """
        Args:
            dim(int): the dimension of samples.
            fixed_std (bool): if True, will use 1 as std for all dimensions.
        """
        super(GaussianDistribution, self).__init__(name)
        self.dim = dim
        self.fixed_std = fixed_std

    def _loglikelihood(self, x, theta):
        eps = 1e-8

        if self.fixed_std:
            mean = theta
            stddev = tf.ones_like(mean)
            exponent = (x - mean)
        else:
            mean, stddev = tf.split(theta, 2, axis=1)
            exponent = (x - mean) / (stddev + eps)

        return tf.reduce_sum(
            - 0.5 * np.log(2 * np.pi) - tf.log(stddev + eps) - 0.5 * tf.square(exponent), 1
        )

    def _encoder_activation(self, dist_param):
        if self.fixed_std:
            return dist_param
        else:
            mean, stddev = tf.split(dist_param, 2, axis=1)
            stddev = tf.exp(stddev)  # just make it positive and assume it's stddev
            # OpenAI code assumes exp(input) is variance. https://github.com/openai/InfoGAN.
            # not sure if there is any theory about this.
            return tf.concat([mean, stddev], axis=1)

    def _sample(self, batch_size, theta):
        if self.fixed_std:
            mean = theta
            stddev = 1
        else:
            mean, stddev = tf.split(theta, 2, axis=1)
        e = tf.random_normal(tf.shape(mean))
        return tf.add(mean, e * stddev, name='sample')

    @property
    def param_dim(self):
        if self.fixed_std:
            return self.dim
        else:
            return 2 * self.dim

    @property
    def sample_dim(self):
        return self.dim


class ProductDistribution(Distribution):
    """A product of a list of independent distributions. """
    def __init__(self, name, dists):
        """
        Args:
            dists(list): list of :class:`Distribution`.
        """
        super(ProductDistribution, self).__init__(name)
        self.dists = dists

    @property
    def param_dim(self):
        return np.sum([d.param_dim for d in self.dists])

    @property
    def sample_dim(self):
        return np.sum([d.sample_dim for d in self.dists])

    def _splitter(self, s, param):
        """Input is split into a list of chunks according
            to dist.param_dim along axis=1

        Args:
            s (tf.Tensor): batch of vectors with shape (batch, param_dim or sample_dim)
            param (bool): split params, otherwise split samples

        Yields:
            tf.Tensor: chunk from input of length N_i with sum N_i = N
        """
        offset = 0
        for dist in self.dists:
            if param:
                off = dist.param_dim
            else:
                off = dist.sample_dim

            yield s[:, offset:offset + off]
            offset += off

    def entropy(self, x, theta):
        """
        Note:
            It returns a list, as one might use different weights for each
            distribution.

        Returns:
            list[tf.Tensor]: entropy of each distribution.
        """
        ret = []
        for dist, xi, ti in zip(self.dists,
                                self._splitter(x, False),
                                self._splitter(theta, True)):
            ret.append(dist.entropy(xi, ti))
        return ret

    def _encoder_activation(self, dist_params):
        rsl = []
        for dist, dist_param in zip(self.dists, self._splitter(dist_params, True)):
            if dist.param_dim > 0:
                rsl.append(dist._encoder_activation(dist_param))
        return tf.concat(rsl, 1)

    def _sample(self, batch_size, theta):
        ret = []
        for dist, ti in zip(self.dists, self._splitter(theta, True)):
            ret.append(dist._sample(batch_size, ti))
        return tf.concat(ret, 1, name='sample')
