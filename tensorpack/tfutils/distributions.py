import tensorflow as tf
from functools import wraps
import numpy as np
from ..utils import logger

__all__ = ['CategoricalDistribution', 'UniformDistribution',
           'NoiseDistribution', 'ProductDistribution']


def class_scope(method):
    """Enhance TensorBoard graph visualization by grouping operators.

    If class of method has member "name", then this is used for name-scoping.

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

    @class_scope
    def loglikelihood(self, x, theta):
        return self._loglikelihood(x, theta)

    @class_scope
    def mutual_information(self, s, x):
        return self._mutual_infomation(s, x)

    def _mutual_information(self, s, x):
        """Approximates mutual information.

        I(s;x) = H(s) - H(s|x)
               = entropy - cross_entropy
               >= H(s) + IE_(p(s,x)) log q(s|x)

        Remarks:
            Think about this as a results "x" from a coin-toss with side "s":
            Mutial information represents all information about "s" given an observation x.
            We just return a lower-bound as an approximation.

        Args:
            s (tf.Tensor): unobserved information
            x (tf.Tensor): observed information

        Returns:
            tf.Tensor: approximated mutual information
        """
        entr = self.entropy(s)
        cross_entr = self.cross_entropy(s, x)
        return tf.subtract(entr, cross_entr, name="mi_%s" % self.name)

    @class_scope
    def entropy(self, p):
        """Estimate of entropy from batch of samples (as average).

        H(p) = IE[I(p)] = - sum( p(p_i) log p(p_i) )
             ~ mean (negative loglikelihood)

        Args:
            p (tf.Tensor): samples from batch

        Returns:
            tf.Tensor: estimated entropy
        """
        return tf.reduce_mean(-self.loglikelihood_prior(p), name="entropy")

    @class_scope
    def cross_entropy(self, p, q):
        """Estimate of cross-entropy from batch of samples.

        H(p, q) = - sum( p(x_i) log q(x_i) )

        Args:
            p (tf.Tensor): Description
            q (tf.Tensor): Description

        Returns:
            tf.Tensor: estimated cross-entropy
        """
        return tf.reduce_mean(-self.loglikelihood(p, q), name="cross_entropy")

    @class_scope
    def loglikelihood_prior(self, x):
        s = self._prior()
        return self._loglikelihood(x, s)

    @class_scope
    def prior(self):
        """Return prior of distribution or anything getting reasonable
        results.

        Returns:
            TYPE: Description
        """
        return self._prior()

    @class_scope
    def model_param(self, dist_param):
        return self._model_param(dist_param)

    def sample(self, batch_size, name="zc"):
        s = self._sample(batch_size, name)
        return s

    def param_dim(self):
        """Distribution parameters require different memory spaces.
        """
        raise NotImplementedError

    def input_dim(self):
        """Distribution parameters require different memory spaces.
        """
        raise NotImplementedError

    def _loglikelihood(self, x, theta):
        raise NotImplementedError

    def _prior(self):
        raise NotImplementedError

    def _sample(self, batch_size, name="zc"):
        raise NotImplementedError

    def _model_param(self, dist_param):
        return dist_param


class CategoricalDistribution(Distribution):
    """Represent categorical distribution

    Attributes:
        cardinality (int): number of categories
    """
    def __init__(self, name, cardinality):
        super(CategoricalDistribution, self).__init__(name)
        self.cardinality = cardinality

    def _loglikelihood(self, x, theta):
        eps = 1e-8
        return tf.reduce_sum(tf.log(theta + eps) * x, reduction_indices=1)

    def _prior(self):
        return tf.constant([1.0 / self.cardinality] * self.cardinality)

    def _sample(self, batch_size, name="zc"):
        ids = tf.multinomial(tf.zeros([batch_size, self.cardinality]), num_samples=1)[:, 0]
        zc = tf.one_hot(ids, self.cardinality)
        return zc

    def _model_param(self, dist_param):
        return tf.nn.softmax(dist_param)

    def param_dim(self):
        return self.cardinality

    def input_dim(self):
        return self.cardinality


class UniformDistribution(Distribution):
    """Represent uniform distribution U(-1,1).

    Remarks:
        There is not such a thing as uniform distributed real numbers.
        Hence, we model prior and posterior as a Gaussian

    Attributes:
        dim (int): dimension
    """
    def __init__(self, name, dim, fixed_std=True):
        super(UniformDistribution, self).__init__(name)
        self.dim = dim
        self.fixed_std = fixed_std

    def _loglikelihood(self, x, theta):
        eps = 1e-8
        # TODO: move to _model_param
        # two cases of theta:
        # - entropy: theta (4,)
        # - cross-entr: theta (?, 4)

        if self.fixed_std:
            mean = theta
            stddev = tf.ones_like(mean)
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

    def _prior(self):
        if self.fixed_std:
            return tf.zeros([self.param_dim()])
        else:
            return tf.concat_v2([tf.zeros([self.param_dim()]), tf.ones([self.param_dim()])], 1, name=name)

    def _sample(self, batch_size, name="zc"):
        zc = tf.random_uniform([batch_size, self.dim], -1, 1)
        return zc

    def _model_param(self, dist_param):
        return dist_param

    def param_dim(self):
        # [mu, sigma]
        if self.fixed_std:
            return self.dim
        else:
            return 2 * self.dim

    def input_dim(self):
        return self.dim


class NoiseDistribution(Distribution):
    """This is not really a distribution, but we implement it as a factor
    without model parameters to simply the actual code.
    """
    def __init__(self, name, dim):
        super(NoiseDistribution, self).__init__(name)
        self.dim = dim

    def _loglikelihood(self, x, theta):
        return 0

    def _prior(self):
        return 0

    def _sample(self, batch_size, name="zc"):
        zc = tf.random_uniform([batch_size, self.dim], -1, 1)
        return zc

    def _model_param(self, dist_param):
        return 0

    def param_dim(self):
        return 0

    def input_dim(self):
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

    def splitter(self, s, output=False):
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
                off = dist.input_dim()

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
        for dist, si, xi in zip(self.dists, self.splitter(s, False), self.splitter(x)):
            if dist.param_dim() > 0:
                MIs.append(dist._mutual_information(si, xi))
        return MIs

    def sample(self, batch_size, name="z_full"):
        """Sample from all factors.

        Args:
            batch_size (int): number of samples

        Remarks:
            This also creates placeholder allowing for inference with custom input.

        Returns:
            tf.Placeholder: placeholder with a default of samples
        """
        samples = []
        for k, dist in enumerate(self.dists):
            init = dist._sample(batch_size)
            plh = tf.placeholder_with_default(init, [batch_size, dist.input_dim()], name='z_' + dist.name)
            samples.append(plh)
            logger.info("Placeholder for %s(%s) is %s " % (dist.name, dist.__class__.__name__, plh.name[:-2]))
        return tf.concat_v2(samples, 1, name=name)

    @class_scope
    def model_param(self, dist_params, name="model_param"):
        rsl = []
        for dist, dist_param in zip(self.dists, self.splitter(dist_params)):
            if dist.param_dim() > 0:
                rsl.append(dist.model_param(dist_param))
        return tf.concat_v2(rsl, 1, name=name)
