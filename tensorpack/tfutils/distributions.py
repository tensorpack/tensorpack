import tensorflow as tf
from functools import wraps
import numpy as np
from ..utils import logger

__all__ = ['Distribution',
           'CategoricalDistribution', 'UniformDistribution',
           'NoiseDistribution', 'ProductDistribution']


def class_scope(method):
    """Enhance TensorBoard graph visualization by grouping operators.

    If class of method has a member "name", then this is used for name-scoping. This groups
    multiple operations at the graph view in Tensorboard.

    Args:
        method (function): method from a python class

    Remarks:
        This is just syntatic sugar to prevent wrinting:
            with tf.name_scope(...):
                ...
        in each method.

    Returns:
        TYPE: results of given function
    """
    @wraps(method)
    def _impl(self, *method_args, **method_kwargs):
        # is there specific name_scope?
        if hasattr(self, "name"):
            distr_name = self.name
        else:
            distr_name = self.__class__.__name__
        # only when it is not already scoped with current class, we scope it
        # TODO: remove this ugly hack, but there is currently no other way
        if distr_name not in tf.no_op(name='.').name[:-1]:
            with tf.name_scope(distr_name + "_" + method.__name__):
                return method(self, *method_args, **method_kwargs)
        else:
            return method(self, *method_args, **method_kwargs)
    return _impl


class Distribution(object):
    """ Base class of symbolic distribution utilities (the distrbution
    parameters can be symbolic tensors). """
    def __init__(self, name):
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
            x.get_shape()[1] == self.sample_dim(), \
            x.get_shape()
        assert theta.get_shape().ndims == 2 and \
            theta.get_shape()[1] == self.param_dim(), \
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
        return self._mutual_infomation(x, theta)

    def _mutual_information(self, x, theta):
        entr = self.prior_entropy(x)
        cross_entr = self.entropy(x, theta)
        return tf.subtract(entr, cross_entr, name="mutual_information")

    @class_scope
    def prior_entropy(self, x):
        r"""
        Estimated entropy from a batch of samples (as average), where the
        likelihood of samples is estimated using the prior distribution.

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
        r""" Entropy of a batch of samples, sampled from this distribution
            parameterized by theta.

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
            a (Batch, param_dim) 2D tensor, containing priors of
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

    def sample_prior(self, batch_size, name="zc"):
        """
        Sample a batch of data with the prior distribution.

        Args:
            batch_size(int):

        Returns:
            samples of shape (batch, sample_dim)
        """
        s = self._sample_prior(batch_size, name)
        return s

    def param_dim(self):
        """
        Returns:
            int: the dimension of parameters of this distribution.
        """
        raise NotImplementedError

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

    def _sample_prior(self, batch_size, name="zc"):
        raise NotImplementedError

    def _encoder_activation(self, dist_param):
        return dist_param


class CategoricalDistribution(Distribution):
    """Represent categorical distribution.

    Attributes:
        cardinality (int): number of categories
    """
    def __init__(self, name, cardinality):
        super(CategoricalDistribution, self).__init__(name)
        self.cardinality = cardinality

    def _loglikelihood(self, x, theta):
        eps = 1e-8
        return tf.reduce_sum(tf.log(theta + eps) * x, reduction_indices=1)

    def _prior(self, batch_size):
        return tf.ones([batch_size, self.cardinality]) * (1.0 / self.cardinality)

    def _sample_prior(self, batch_size, name="zc"):
        ids = tf.multinomial(tf.zeros([batch_size, self.cardinality]), num_samples=1)[:, 0]
        zc = tf.one_hot(ids, self.cardinality)
        return zc

    def _encoder_activation(self, dist_param):
        return tf.nn.softmax(dist_param)

    def param_dim(self):
        return self.cardinality

    def sample_dim(self):
        return self.cardinality


class UniformDistribution(Distribution):
    """Represent uniform distribution U(-1,1).

    Remarks:
        There is not such a thing as uniformly distributed real numbers.
        Hence, this implements a Gaussian with uniform sample_prior.

    Attributes:
        dim (int): dimension
    """
    def __init__(self, name, dim, fixed_std=True):
        super(UniformDistribution, self).__init__(name)
        self.dim = dim
        self.fixed_std = fixed_std

    def _loglikelihood(self, x, theta):
        eps = 1e-8
        # TODO: move to _encoder_activation
        # two cases of theta:
        # - entropy: theta (4,)
        # - cross-entr: theta (?, 4)

        if self.fixed_std:
            mean = theta
            stddev = tf.ones_like(mean)
            exponent = (x - mean)
        else:
            if len(theta.get_shape()) == 1:
                l = theta.get_shape()[0] // 2
                mean = theta[:l]
                stddev = theta[l:self.dim * 2]
            else:
                l = theta.get_shape()[1] // 2
                mean = theta[:, 0:l]
                stddev = theta[:, l:self.dim * 2]

            stddev = tf.sqrt(tf.exp(stddev) + eps)
            exponent = (x - mean) / (stddev + eps)

        return tf.reduce_sum(
            - 0.5 * np.log(2 * np.pi) - tf.log(stddev + eps) - 0.5 * tf.square(exponent),
            reduction_indices=1
        )

    def _prior(self, batch_size):
        if self.fixed_std:
            return tf.zeros([batch_size, self.param_dim()])
        else:
            return tf.concat_v2([tf.zeros([batch_size, self.param_dim()]),
                                 tf.ones([batch_size, self.param_dim()])], 1)

    def _sample_prior(self, batch_size, name="zc"):
        return tf.random_uniform([batch_size, self.dim], -1, 1)

    def _encoder_activation(self, dist_param):
        return dist_param

    def param_dim(self):
        if self.fixed_std:
            return self.dim
        else:
            return 2 * self.dim

    def sample_dim(self):
        return self.dim


class NoiseDistribution(Distribution):
    """This is not really a distribution, but we implement it as a factor
    without model parameters to simplify the actual code.
    """
    def __init__(self, name, dim):
        super(NoiseDistribution, self).__init__(name)
        self.dim = dim

    def _loglikelihood(self, x, theta):
        return 0

    def _prior(self):
        return 0

    def _sample_prior(self, batch_size, name="zc"):
        zc = tf.random_uniform([batch_size, self.dim], -1, 1)
        return zc

    def _encoder_activation(self, dist_param):
        return 0

    def param_dim(self):
        return 0

    def sample_dim(self):
        return self.dim


class ProductDistribution(Distribution):
    """Represent a product distribution"""
    def __init__(self, name, dists):
        super(ProductDistribution, self).__init__(name)
        self.dists = dists

    def param_dim(self):
        """Number of estimated parameters required from discriminator.

        Remarks:
            Some distribution like d-dim Gaussian require 2*d parameters
            for mean and variance.

        Returns:
            int: required parameters
        """
        return np.sum([d.param_dim() for d in self.dists])

    def _splitter(self, s, output=False):
        """Input is split into list of chunks according dist.param_dim() along 2-axis

        Args:
            s (tf.Tensor): batch of vectors with shape BxN

        Yields:
            tf.Tensor: chunk from input of length N_i with sum N_i = N
        """
        offset = 0
        for dist in self.dists:
            if output:
                off = dist.param_dim()
            else:
                off = dist.sample_dim()

            yield s[:, offset:offset + off]
            offset += off

    def mutual_information(self, s, x):
        """Return mutual information of all factors but skip noise.

        Args:
            s (tf.Tensor): unobserved information
            x (tf.Tensor): observed information

        Remarks:
            This returns a list, as one might use different weights for each factor.

        Returns:
            list(tf.Tensor): all mutual informations
        """
        MIs = []  # noqa
        for dist, si, xi in zip(self.dists, self._splitter(s, False), self._splitter(x)):
            if dist.param_dim() > 0:
                MIs.append(dist._mutual_information(si, xi))
        return MIs

    def sample_prior(self, batch_size, name="z_full"):
        """Sample from all factors.

        Args:
            batch_size (int): number of samples

        Remarks:
            This also creates placeholder allowing to do inference with custom input.

        Returns:
            tf.Placeholder: placeholder with a default of samples
        """
        samples = []
        for k, dist in enumerate(self.dists):
            init = dist._sample_prior(batch_size)
            plh = tf.placeholder_with_default(init, [batch_size, dist.sample_dim()], name='z_' + dist.name)
            samples.append(plh)
            logger.info("Placeholder for %s(%s) is %s " % (dist.name, dist.__class__.__name__, plh.name[:-2]))
        return tf.concat_v2(samples, 1, name=name)

    @class_scope
    def encoder_activation(self, dist_params, name="encoder_activation"):
        rsl = []
        for dist, dist_param in zip(self.dists, self._splitter(dist_params)):
            if dist.param_dim() > 0:
                rsl.append(dist.encoder_activation(dist_param))
        return tf.concat_v2(rsl, 1, name=name)
