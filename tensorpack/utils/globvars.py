#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: globvars.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import six
import argparse
from . import logger

__all__ = ['globalns', 'GlobalNS']

if six.PY2:
    class NS:
        pass
else:
    import types
    NS = types.SimpleNamespace


# TODO make it singleton

class GlobalNS(NS):
    """
    The class of the globalns instance.
    """
    def use_argument(self, args):
        """
        Add the content of :class:`argparse.Namespace` to this ns.

        Args:
            args (argparse.Namespace): arguments
        """
        assert isinstance(args, argparse.Namespace), type(args)
        for k, v in six.iteritems(vars(args)):
            if hasattr(self, k):
                logger.warn("Attribute {} in globalns will be overwritten!")
            setattr(self, k, v)


globalns = GlobalNS()
"""
A namespace to store global variables.

Examples:

.. code-block:: none

    import tensorpack.utils.globalns as G

    G.depth = 18
    G.batch_size = 1
    G.use_argument(parser.parse_args())
"""
