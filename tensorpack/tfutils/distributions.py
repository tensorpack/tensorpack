import tensorflow as tf
from functools import wraps
import numpy as np
from ..utils import logger
from ..tfutils import get_name_scope_name

__all__ = ['Distribution',
           'CategoricalDistribution', 'GaussianDistributionUniformPrior',
           'NoiseDistribution', 'ProductDistribution']

# TODO encoder_activation and the ProductDistribution class brings many redundant concat and split


def class_scope(func):
    """
    A decorator which wraps a function with a name_scope: "{class_name}_{method_name}".
    The "{class_name}" is either ``cls.name`` or simply the class name.
    It helps enhance TensorBoard graph visualization by grouping operators.

    This is just syntatic sugar to prevent wrinting: with
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
    (the distrbution parameters can be symbolic tensors).
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
    def loglikelihood_prior(self, x):
        """likelihood from prior for this distribution

        Args:
            x: samples of shape (batch, sample_dim)

        Returns:
            a symbolic vector containing loglikelihood of each sample,
            using prior of this distribution.
        """
        assert x.get_shape().ndims == 2 and \
            x.get_shape()[1] == self.sample_dim, \
            x.get_shape()
        batch_size = x.get_shape().as_list()[0]
        s = self.prior(batch_size)
        return self._loglikelihood(x, s)

    @class_scope
    def mutual_information(self, x, theta):
        """
        Approximates mutual information between x and some information s.
        Here we return a variational lower bound of the mutual information,
        assuming a proposal distribution Q(x|s) (which approximates P(x|s) )
        has the form of this distribution parameterized by theta.


        .. math::

            I(x;s) = H(x) - H(x|s)
                   = H(x) + E[\log P(x|s)]
                   \\ge H(x) + E_{x \sim P(x|s)}[\log Q(x|s)]

        Args:
            x: samples of shape (batch, sample_dim)
            theta: parameters defining the proposal distribution Q. shape (batch, param_dim).

        Returns:
            lower-bounded mutual information, a scalar tensor.
        """

        entr = self.prior_entropy(x)
        cross_entr = self.entropy(x, theta)
        return tf.subtract(entr, cross_entr, name="mutual_information")

    @class_scope
    def prior_entropy(self, x):
        r"""
        Estimated entropy of the prior distribution,
        from a batch of samples (as average). It
        estimates the likelihood of samples using the prior distribution.

        .. math::

            H(x) = -E[\log p(x_i)], \text{where } p \text{ is the prior}

        Args:
            x: samples of shape (batch, sample_dim)

        Returns:
            a scalar, estimated entropy.
        """
        return tf.reduce_mean(-self.loglikelihood_prior(x), name="prior_entropy")

    @class_scope
    def entropy(self, x, theta):
        r""" Entropy of this distribution parameterized by theta,
            esimtated from a batch of samples.

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
    def prior(self, batch_size):
        """Get the prior parameters of this distribution.

        Returns:
            a (batch, param_dim) 2D tensor, containing priors of
            this distribution repeated for batch_size times.
        """
        return self._prior(batch_size)

    @class_scope
    def encoder_activation(self, dist_param):
        """ An activation function to produce
            feasible distribution parameters from unconstrained raw network output.

        Args:
            dist_param: output from a network, of shape (batch, param_dim).

        Returns:
            a tensor of the same shape, the distribution parameters.
        """
        return self._encoder_activation(dist_param)

    def sample_prior(self, batch_size):
        """
        Sample a batch of data with the prior distribution.

        Args:
            batch_size(int):

        Returns:
            samples of shape (batch, sample_dim)
        """
        s = self._sample_prior(batch_size)
        return s

    @property
    def param_dim(self):
        """
        Returns:
            int: the dimension of parameters of this distribution.
        """
        raise NotImplementedError

    @property
    def sample_dim(self):
        """
        Returns:
            int: the dimension of samples out of this distribution.
        """
        raise NotImplementedError

    def _loglikelihood(self, x, theta):
        raise NotImplementedError

    def _prior(self, batch_size):
        raise NotImplementedError

    def _sample_prior(self, batch_size):
        raise NotImplementedError

    def _encoder_activation(self, dist_param):
        return dist_param


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
        return tf.reduce_sum(tf.log(theta + eps) * x, reduction_indices=1)

    def _prior(self, batch_size):
        return tf.constant(1.0 / self.cardinality,
                           tf.float32, [batch_size, self.cardinality])

    def _sample_prior(self, batch_size):
        ids = tf.multinomial(tf.zeros([batch_size, self.cardinality]), num_samples=1)[:, 0]
        ret = tf.one_hot(ids, self.cardinality)
        return ret

    def _encoder_activation(self, dist_param):
        return tf.nn.softmax(dist_param)

    @property
    def param_dim(self):
        return self.cardinality

    @property
    def sample_dim(self):
        return self.cardinality


class GaussianDistributionUniformPrior(Distribution):
    """Gaussian distribution with prior U(-1,1).
    It implements a Gaussian with uniform :meth:`sample_prior` method.
    """
    def __init__(self, name, dim, fixed_std=True):
        """
        Args:
            dim(int): the dimension of samples.
            fixed_std (bool): if True, will use 1 as std for all dimensions.
        """
        super(GaussianDistributionUniformPrior, self).__init__(name)
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
            - 0.5 * np.log(2 * np.pi) - tf.log(stddev + eps) - 0.5 * tf.square(exponent),
            reduction_indices=1
        )

    def _prior(self, batch_size):
        if self.fixed_std:
            return tf.zeros([batch_size, self.param_dim])
        else:
            return tf.concat_v2([tf.zeros([batch_size, self.param_dim]),
                                 tf.ones([batch_size, self.param_dim])], 1)

    def _sample_prior(self, batch_size):
        return tf.random_uniform([batch_size, self.dim], -1, 1)

    def _encoder_activation(self, dist_param):
        if self.fixed_std:
            return dist_param
        else:
            mean, stddev = tf.split(dist_param, 2, axis=1)
            # this is from https://github.com/openai/InfoGAN. don't know why
            stddev = tf.sqrt(tf.exp(stddev))
            return tf.concat_v2([mean, stddev], axis=1)

    @property
    def param_dim(self):
        if self.fixed_std:
            return self.dim
        else:
            return 2 * self.dim

    @property
    def sample_dim(self):
        return self.dim


class NoiseDistribution(Distribution):
    """This is not really a distribution.
    It is the uniform noise input of GAN which shares interface with Distribution, to
    simplify implementation of GAN.
    """
    def __init__(self, name, dim):
        """
        Args:
            dim(int): the dimension of the noise.
        """
        # TODO more options, e.g. use gaussian or uniform?
        super(NoiseDistribution, self).__init__(name)
        self.dim = dim

    def _loglikelihood(self, x, theta):
        return 0

    def _prior(self):
        return 0

    def _sample_prior(self, batch_size):
        zc = tf.random_uniform([batch_size, self.dim], -1, 1)
        return zc

    def _encoder_activation(self, dist_param):
        return 0

    @property
    def param_dim(self):
        return 0

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

    def mutual_information(self, x, theta):
        """
        Return mutual information of all distributions but skip the
        unparameterized ones.

        Note:
            It returns a list, as one might use different weights for each
            distribution.

        Returns:
            list[tf.Tensor]: mutual informations of each distribution.
        """
        MIs = []  # noqa
        for dist, xi, ti in zip(self.dists,
                                self._splitter(x, False),
                                self._splitter(theta, True)):
            if dist.param_dim > 0:
                MIs.append(dist.mutual_information(xi, ti))
        return MIs

    def sample_prior(self, batch_size, name='sample_prior'):
        """
        Concat the samples from all distributions.

        Returns:
            tf.Tensor: a tensor of shape (batch, sample_dim), but first dimension is statically unknown,
                allowing you to do inference with custom batch size.
        """
        samples = []
        for k, dist in enumerate(self.dists):
            init = dist._sample_prior(batch_size)
            plh = tf.placeholder_with_default(init, [batch_size, dist.sample_dim], name='z_' + dist.name)
            samples.append(plh)
            logger.info("Placeholder for %s(%s) is %s " % (dist.name, dist.__class__.__name__, plh.name[:-2]))
        return tf.concat_v2(samples, 1, name=name)

    def _encoder_activation(self, dist_params):
        rsl = []
        for dist, dist_param in zip(self.dists, self._splitter(dist_params, True)):
            if dist.param_dim > 0:
                rsl.append(dist._encoder_activation(dist_param))
        return tf.concat_v2(rsl, 1)
