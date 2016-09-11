#!/bin/bash -e
# File: update.sh
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

make clean
sphinx-apidoc -o modules ../tensorpack -f -d 10
make html
