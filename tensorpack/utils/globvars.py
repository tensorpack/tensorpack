#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: globvars.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import six
import argparse

__all__ = ['globalns', 'use_global_argument']

if six.PY2:
    class NS:
        pass
else:
    import types
    NS = types.SimpleNamespace

globalns = NS()


def use_global_argument(args):
    """
    Add the content of :class:`argparse.Namespace` to globalns.

    Args:
        args (argparse.Namespace): arguments
    """
    assert isinstance(args, argparse.Namespace), type(args)
    for k, v in six.iteritems(vars(args)):
        setattr(globalns, k, v)
