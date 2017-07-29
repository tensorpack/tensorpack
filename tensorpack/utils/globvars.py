#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: globvars.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import six
import argparse

__all__ = ['globalns']

if six.PY2:
    class NS:
        pass
else:
    import types
    NS = types.SimpleNamespace


class MyNS(NS):
    def use_argument(self, args):
        """
        Add the content of :class:`argparse.Namespace` to this ns.

        Args:
            args (argparse.Namespace): arguments
        """
        assert isinstance(args, argparse.Namespace), type(args)
        for k, v in six.iteritems(vars(args)):
            setattr(self, k, v)


globalns = MyNS()
