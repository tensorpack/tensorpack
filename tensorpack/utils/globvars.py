#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: globvars.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import six

__all__ = ['globalns']

if six.PY2:
    class NS: pass
else:
    import types
    NS = types.SimpleNamespace

globalns = NS()
