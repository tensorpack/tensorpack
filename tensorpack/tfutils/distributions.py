import tensorflow as tf
from functools import wraps
import numpy as np
from ..utils import logger

__all__ = ['CategoricalDistribution', 'UniformDistribution']


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
        return tf.subtract(entr, cross_entr)

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
        """Return prior of distribution and anthing getting reasonable
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
        logger.info("sample of distribution '%s' (%s) has name '%s'" % (self.name, self.__class__.__name__,
                                                                        s.name[:-2]))
        return s

    def size(self):
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
        zc = tf.placeholder_with_default(zc, [None, self.cardinality], name=name)
        return zc

    def _model_param(self, dist_param):
        return tf.nn.softmax(dist_param)

    def size(self):
        return self.cardinality


class UniformDistribution(Distribution):
    """Represent uniform distribution U(-1,1).

    Remarks:
        There is not such a thing as uniform distributed real numbers.
        Hence, we model prior and posterior aus a Gaussian

    Attributes:
        dim (int): dimension
    """
    def __init__(self, name, dim):
        super(UniformDistribution, self).__init__(name)
        self.dim = dim

    def _loglikelihood(self, x, theta):
        eps = 1e-8
        # TODO: move to _model_param
        # two cases of theta:
        # - entropy: theta (4,)
        # - cross-entr: theta (?, 4)
        if len(theta.get_shape()) == 1:
            l = theta.get_shape()[0] // 2
            mean = theta[:l]
            stddev = theta[l:self.dim * 2]
        else:
            l = theta.get_shape()[1] // 2
            mean = theta[:, 0:l]
            stddev = theta[:, l:self.dim * 2]

        # only predict mean
        stddev = tf.ones_like(stddev)
        exponent = (x - mean) / (stddev + eps)

        return tf.reduce_sum(
            - 0.5 * np.log(2 * np.pi) - tf.log(stddev + eps) - 0.5 * tf.square(exponent),
            reduction_indices=1
        )

    def _prior(self):
        return tf.random_uniform([self.dim * 2], minval=-1., maxval=1.)

    def _sample(self, batch_size, name="zc"):
        zc = tf.random_uniform([batch_size, self.dim], -1, 1)
        zc = tf.placeholder_with_default(zc, [None, self.dim], name=name)
        return zc

    def _model_param(self, dist_param):
        return dist_param

    def size(self):
        # [mu, sigma]
        return 2 * self.dim


class ProductDistribution(object):
    """docstring for ProductDistribution (see Product-Space)"""
    def __init__(self, name, dists):
        super(ProductDistribution, self).__init__(name)
        self.dists = dists

    def size(self):
        return np.sum([d.size() for d in self.dists])

    def splitter(self, s):
        """Input is split into list of chunks according dist.size() along 2-axis

        Args:
            s (tf.Tensor): batch of vectors with shape BxN

        Yields:
            tf.Tensor: chunk from input of length N_i with sum N_i = N
        """
        offset = 0
        for dist in self.dists:
            yield s[:, offset:offset + dist.size()]
            offset += dist.size()

    @class_scope
    def mutual_information(self, ss, xs):
        """Summary
        
        Args:
            ss (TYPE): Description
            xs (TYPE): Description
        
        Returns:
            TYPE: Description
        """
        MIs = []  # noqa
        with tf.name_scope("mutual_information"):
            for dist, s, x in zip(self.dists, self.splitter(ss), self.splitter(xs)):
                MIs.append(dist.mutual_information(s, x))

            return tf.add_n(MIs, name="total")

    def sample(self, batch_size, names=None):
        for k, dist in enumerate(self.dists):
            name = 1 if names is not None else names[k]
            z = dists.sample(batch_size, name=name)

        s = self._sample(batch_size, name)
        logger.info("sample of distribution '%s' (%s) has name '%s'" % (self.name, self.__class__.__name__,
                                                                        s.name[:-2]))
        return s
