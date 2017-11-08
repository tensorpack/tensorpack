#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: interface.py


__all__ = ['launch_train_with_config']


def launch_train_with_config(config, trainer):
    from ..train.interface import launch_train_with_config as old_launch
    old_launch(config, trainer)
